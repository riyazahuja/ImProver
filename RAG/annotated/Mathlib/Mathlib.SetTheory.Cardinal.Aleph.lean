/-- An ordinal is initial when it is the first ordinal with a given cardinality.

This is written as `o.card.ord = o`, i.e. `o` is the smallest ordinal with cardinality `o.card`. -/
def IsInitial (o : Ordinal) : Prop :=
  o.card.ord = o


theorem IsInitial.ord_card {o : Ordinal} (h : IsInitial o) : o.card.ord = o := h


theorem IsInitial.card_le_card {a b : Ordinal} (ha : IsInitial a) : a.card ≤ b.card ↔ a ≤ b := by
  /-
    a b : Ordinal.{u_1}
    ha : a.IsInitial
    ⊢ Iff (LE.le a.card b.card) (LE.le a b)
  -/
  refine ⟨fun h ↦ ?_, Ordinal.card_le_card⟩
  /-
    a b : Ordinal.{u_1}
    ha : a.IsInitial
    h : LE.le a.card b.card
    ⊢ LE.le a b
  -/
  rw [← ord_le_ord, ha.ord_card] at h
  /-
    a b : Ordinal.{u_1}
    ha : a.IsInitial
    h : LE.le a b.card.ord
    ⊢ LE.le a b
  -/
  exact h.trans (ord_card_le b)
  /-
    🎉 no goals
  -/


theorem IsInitial.card_lt_card {a b : Ordinal} (hb : IsInitial b) : a.card < b.card ↔ a < b :=
  lt_iff_lt_of_le_iff_le hb.card_le_card


theorem isInitial_ord (c : Cardinal) : IsInitial c.ord := by
  /-
    c : Cardinal.{u_1}
    ⊢ c.ord.IsInitial
  -/
  rw [IsInitial, card_ord]
  /-
    🎉 no goals
  -/


theorem isInitial_natCast (n : ℕ) : IsInitial n := by
  /-
    n : Nat
    ⊢ (↑n).IsInitial
  -/
  rw [IsInitial, card_nat, ord_nat]
  /-
    🎉 no goals
  -/


theorem isInitial_zero : IsInitial 0 := by
  /-
    ⊢ Ordinal.IsInitial 0
  -/
  exact_mod_cast isInitial_natCast 0
  /-
    🎉 no goals
  -/


theorem isInitial_one : IsInitial 1 := by
  /-
    ⊢ Ordinal.IsInitial 1
  -/
  exact_mod_cast isInitial_natCast 1
  /-
    🎉 no goals
  -/


theorem isInitial_omega0 : IsInitial ω := by
  /-
    ⊢ Ordinal.omega0.IsInitial
  -/
  rw [IsInitial, card_omega0, ord_aleph0]
  /-
    🎉 no goals
  -/


theorem not_bddAbove_isInitial : ¬ BddAbove {x | IsInitial x} := by
  /-
    ⊢ Not (BddAbove (setOf fun x => x.IsInitial))
  -/
  rintro ⟨a, ha⟩
  /-
    case intro
    a : Ordinal.{u_1}
    ha : Membership.mem (upperBounds (setOf fun x => x.IsInitial)) a
    ⊢ False
  -/
  have := ha (isInitial_ord (succ a.card))
  /-
    case intro
    a : Ordinal.{u_1}
    ha : Membership.mem (upperBounds (setOf fun x => x.IsInitial)) a
    this : LE.le (Order.succ a.card).ord a
    ⊢ False
  -/
  rw [ord_le] at this
  /-
    case intro
    a : Ordinal.{u_1}
    ha : Membership.mem (upperBounds (setOf fun x => x.IsInitial)) a
    this : LE.le (Order.succ a.card) a.card
    ⊢ False
  -/
  exact (lt_succ _).not_le this
  /-
    🎉 no goals
  -/


/-- Initial ordinals are order-isomorphic to the cardinals. -/
@[simps!]
def isInitialIso : {x // IsInitial x} ≃o Cardinal where
  toFun x := x.1.card
  invFun x := ⟨x.ord, isInitial_ord _⟩
  left_inv x := Subtype.ext x.2.ord_card
  right_inv x := card_ord x
  map_rel_iff' {a _} := a.2.card_le_card


/-- The "pre-omega" function gives the initial ordinals listed by their ordinal index.
`preOmega n = n`, `preOmega ω = ω`, `preOmega (ω + 1) = ω₁`, etc.

For the more common omega function skipping over finite ordinals, see `Ordinal.omega`. -/
def preOmega : Ordinal.{u} ↪o Ordinal.{u} where
  toFun := enumOrd {x | IsInitial x}
  inj' _ _ h := enumOrd_injective not_bddAbove_isInitial h
  map_rel_iff' := enumOrd_le_enumOrd not_bddAbove_isInitial


theorem coe_preOmega : preOmega = enumOrd {x | IsInitial x} :=
  rfl


theorem preOmega_strictMono : StrictMono preOmega :=
  preOmega.strictMono


theorem preOmega_lt_preOmega {o₁ o₂ : Ordinal} : preOmega o₁ < preOmega o₂ ↔ o₁ < o₂ :=
  preOmega.lt_iff_lt


theorem preOmega_le_preOmega {o₁ o₂ : Ordinal} : preOmega o₁ ≤ preOmega o₂ ↔ o₁ ≤ o₂ :=
  preOmega.le_iff_le


theorem preOmega_max (o₁ o₂ : Ordinal) : preOmega (max o₁ o₂) = max (preOmega o₁) (preOmega o₂) :=
  preOmega.monotone.map_max


theorem isInitial_preOmega (o : Ordinal) : IsInitial (preOmega o) :=
  enumOrd_mem not_bddAbove_isInitial o


theorem le_preOmega_self (o : Ordinal) : o ≤ preOmega o :=
  preOmega_strictMono.le_apply


@[simp]
theorem preOmega_zero : preOmega 0 = 0 := by
  /-
    ⊢ Eq (Ordinal.preOmega 0) 0
  -/
  rw [coe_preOmega, enumOrd_zero]
  /-
    ⊢ Eq (InfSet.sInf (setOf fun x => x.IsInitial)) 0
  -/
  exact csInf_eq_bot_of_bot_mem isInitial_zero
  /-
    🎉 no goals
  -/


@[simp]
theorem preOmega_natCast (n : ℕ) : preOmega n = n := by
  induction n with
  | zero => exact preOmega_zero
  | succ n IH =>
    apply (le_preOmega_self _).antisymm'
    apply enumOrd_succ_le not_bddAbove_isInitial (isInitial_natCast _) (IH.trans_lt _)
    rw [Nat.cast_lt]
    exact lt_succ n

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem preOmega_ofNat (n : ℕ) [n.AtLeastTwo] : preOmega (no_index (OfNat.ofNat n)) = n :=
  preOmega_natCast n


theorem preOmega_le_of_forall_lt {o a : Ordinal} (ha : IsInitial a) (H : ∀ b < o, preOmega b < a) :
    preOmega o ≤ a :=
  enumOrd_le_of_forall_lt ha H


theorem isNormal_preOmega : IsNormal preOmega := by
  /-
    ⊢ Ordinal.IsNormal ⇑Ordinal.preOmega
  -/
  rw [isNormal_iff_strictMono_limit]
  refine ⟨preOmega_strictMono, fun o ho a ha ↦
    (preOmega_le_of_forall_lt (isInitial_ord _) fun b hb ↦ ?_).trans (ord_card_le a)⟩
  /-
    o : Ordinal.{u_1}
    ho : o.IsLimit
    a : Ordinal.{u_1}
    ha : ∀ (b : Ordinal.{u_1}), LT.lt b o → LE.le (Ordinal.preOmega b) a
    b : Ordinal.{u_1}
    hb : LT.lt b o
    ⊢ LT.lt (Ordinal.preOmega b) a.card.ord
  -/
  rw [← (isInitial_ord _).card_lt_card, card_ord]
  /-
    o : Ordinal.{u_1}
    ho : o.IsLimit
    a : Ordinal.{u_1}
    ha : ∀ (b : Ordinal.{u_1}), LT.lt b o → LE.le (Ordinal.preOmega b) a
    b : Ordinal.{u_1}
    hb : LT.lt b o
    ⊢ LT.lt (Ordinal.preOmega b).card a.card
  -/
  apply lt_of_lt_of_le _ (card_le_card <| ha _ (ho.succ_lt hb))
  /-
    o : Ordinal.{u_1}
    ho : o.IsLimit
    a : Ordinal.{u_1}
    ha : ∀ (b : Ordinal.{u_1}), LT.lt b o → LE.le (Ordinal.preOmega b) a
    b : Ordinal.{u_1}
    hb : LT.lt b o
    ⊢ LT.lt (Ordinal.preOmega b).card (Ordinal.preOmega (Order.succ b)).card
  -/
  rw [(isInitial_preOmega _).card_lt_card, preOmega_lt_preOmega]
  /-
    o : Ordinal.{u_1}
    ho : o.IsLimit
    a : Ordinal.{u_1}
    ha : ∀ (b : Ordinal.{u_1}), LT.lt b o → LE.le (Ordinal.preOmega b) a
    b : Ordinal.{u_1}
    hb : LT.lt b o
    ⊢ LT.lt b (Order.succ b)
  -/
  exact lt_succ b
  /-
    🎉 no goals
  -/


@[simp]
theorem range_preOmega : range preOmega = {x | IsInitial x} :=
  range_enumOrd not_bddAbove_isInitial


theorem mem_range_preOmega_iff {x : Ordinal} : x ∈ range preOmega ↔ IsInitial x := by
  /-
    x : Ordinal.{u_1}
    ⊢ Iff (Membership.mem (Set.range ⇑Ordinal.preOmega) x) x.IsInitial
  -/
  rw [range_preOmega, mem_setOf]
  /-
    🎉 no goals
  -/


alias ⟨_, IsInitial.mem_range_preOmega⟩ := mem_range_preOmega_iff


@[simp]
theorem preOmega_omega0 : preOmega ω = ω := by
  /-
    ⊢ Eq (Ordinal.preOmega Ordinal.omega0) Ordinal.omega0
  -/
  simp_rw [← isNormal_preOmega.apply_omega0, preOmega_natCast, iSup_natCast]
  /-
    🎉 no goals
  -/


@[simp]
theorem omega0_le_preOmega_iff {x : Ordinal} : ω ≤ preOmega x ↔ ω ≤ x := by
  /-
    x : Ordinal.{u_1}
    ⊢ Iff (LE.le Ordinal.omega0 (Ordinal.preOmega x)) (LE.le Ordinal.omega0 x)
  -/
  conv_lhs => rw [← preOmega_omega0, preOmega_le_preOmega]
  /-
    🎉 no goals
  -/


@[simp]
theorem omega0_lt_preOmega_iff {x : Ordinal} : ω < preOmega x ↔ ω < x := by
  /-
    x : Ordinal.{u_1}
    ⊢ Iff (LT.lt Ordinal.omega0 (Ordinal.preOmega x)) (LT.lt Ordinal.omega0 x)
  -/
  conv_lhs => rw [← preOmega_omega0, preOmega_lt_preOmega]
  /-
    🎉 no goals
  -/


/-- The `omega` function gives the infinite initial ordinals listed by their ordinal index.
`omega 0 = ω`, `omega 1 = ω₁` is the first uncountable ordinal, and so on.

This is not to be confused with the first infinite ordinal `Ordinal.omega0`.

For a version including finite ordinals, see `Ordinal.preOmega`. -/
def omega : Ordinal ↪o Ordinal :=
  (OrderEmbedding.addLeft ω).trans preOmega


@[inherit_doc]
scoped notation "ω_ " => omega


/-- `ω₁` is the first uncountable ordinal. -/
scoped notation "ω₁" => ω_ 1


theorem omega_eq_preOmega (o : Ordinal) : ω_ o = preOmega (ω + o) :=
  rfl


theorem omega_strictMono : StrictMono omega :=
  omega.strictMono


theorem omega_lt_omega {o₁ o₂ : Ordinal} : ω_ o₁ < ω_ o₂ ↔ o₁ < o₂ :=
  omega.lt_iff_lt


theorem omega_le_omega {o₁ o₂ : Ordinal} : ω_ o₁ ≤ ω_ o₂ ↔ o₁ ≤ o₂ :=
  omega.le_iff_le


theorem omega_max (o₁ o₂ : Ordinal) : ω_ (max o₁ o₂) = max (ω_ o₁) (ω_ o₂) :=
  omega.monotone.map_max


theorem preOmega_le_omega (o : Ordinal) : preOmega o ≤ ω_ o :=
  preOmega_le_preOmega.2 (Ordinal.le_add_left _ _)


theorem isInitial_omega (o : Ordinal) : IsInitial (omega o) :=
  isInitial_preOmega _


theorem le_omega_self (o : Ordinal) : o ≤ omega o :=
  omega_strictMono.le_apply


@[simp]
theorem omega_zero : ω_ 0 = ω := by
  /-
    ⊢ Eq (Ordinal.omega 0) Ordinal.omega0
  -/
  rw [omega_eq_preOmega, add_zero, preOmega_omega0]
  /-
    🎉 no goals
  -/


theorem omega0_le_omega (o : Ordinal) : ω ≤ ω_ o := by
  /-
    o : Ordinal.{u_1}
    ⊢ LE.le Ordinal.omega0 (Ordinal.omega o)
  -/
  rw [← omega_zero, omega_le_omega]
  /-
    o : Ordinal.{u_1}
    ⊢ LE.le 0 o
  -/
  exact Ordinal.zero_le o
  /-
    🎉 no goals
  -/


/-- For the theorem `0 < ω`, see `omega0_pos`. -/
theorem omega_pos (o : Ordinal) : 0 < ω_ o :=
  omega0_pos.trans_le (omega0_le_omega o)


theorem omega0_lt_omega1 : ω < ω₁ := by
  /-
    ⊢ LT.lt Ordinal.omega0 (Ordinal.omega 1)
  -/
  rw [← omega_zero, omega_lt_omega]
  /-
    ⊢ LT.lt 0 1
  -/
  exact zero_lt_one
  /-
    🎉 no goals
  -/


@[deprecated omega0_lt_omega1 (since := "2024-10-11")]
alias omega_lt_omega1 := omega0_lt_omega1


theorem isNormal_omega : IsNormal omega :=
  isNormal_preOmega.trans (isNormal_add_right _)


@[simp]
theorem range_omega : range omega = {x | ω ≤ x ∧ IsInitial x} := by
  /-
    ⊢ Eq (Set.range ⇑Ordinal.omega) (setOf fun x => And (LE.le Ordinal.omega0 x) x …
  -/
  ext x
  /-
    case h
    x : Ordinal.{u_1}
    ⊢ Iff (Membership.mem (Set.range ⇑Ordinal.omega) x) (Membership.mem (setOf fun …
  -/
  constructor
    /-
      case h.mp
      x : Ordinal.{u_1}
      ⊢ Membership.mem (Set.range ⇑Ordinal.omega) x → Membership.mem (setOf fun x => …
    -/
  · rintro ⟨a, rfl⟩
    /-
      case h.mp.intro
      a : Ordinal.{u_1}
      ⊢ Membership.mem (setOf fun x => And (LE.le Ordinal.omega0 x) x.IsInitial) (Or …
    -/
    exact ⟨omega0_le_omega a, isInitial_omega a⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      x : Ordinal.{u_1}
      ⊢ Membership.mem (setOf fun x => And (LE.le Ordinal.omega0 x) x.IsInitial) x → …
    -/
  · rintro ⟨ha', ha⟩
    /-
      case h.mpr.intro
      x : Ordinal.{u_1}
      ha' : LE.le Ordinal.omega0 x
      ha : x.IsInitial
      ⊢ Membership.mem (Set.range ⇑Ordinal.omega) x
    -/
    obtain ⟨a, rfl⟩ := ha.mem_range_preOmega
    /-
      case h.mpr.intro.intro
      a : Ordinal.{u_1}
      ha' : LE.le Ordinal.omega0 (Ordinal.preOmega a)
      ha : (Ordinal.preOmega a).IsInitial
      ⊢ Membership.mem (Set.range ⇑Ordinal.omega) (Ordinal.preOmega a)
    -/
    use a - ω
    /-
      case h
      a : Ordinal.{u_1}
      ha' : LE.le Ordinal.omega0 (Ordinal.preOmega a)
      ha : (Ordinal.preOmega a).IsInitial
      ⊢ Eq (Ordinal.omega (HSub.hSub a Ordinal.omega0)) (Ordinal.preOmega a)
    -/
    rw [omega0_le_preOmega_iff] at ha'
    /-
      case h
      a : Ordinal.{u_1}
      ha' : LE.le Ordinal.omega0 a
      ha : (Ordinal.preOmega a).IsInitial
      ⊢ Eq (Ordinal.omega (HSub.hSub a Ordinal.omega0)) (Ordinal.preOmega a)
    -/
    rw [omega_eq_preOmega, Ordinal.add_sub_cancel_of_le ha']
    /-
      🎉 no goals
    -/


theorem mem_range_omega_iff {x : Ordinal} : x ∈ range omega ↔ ω ≤ x ∧ IsInitial x := by
  /-
    x : Ordinal.{u_1}
    ⊢ Iff (Membership.mem (Set.range ⇑Ordinal.omega) x) (And (LE.le Ordinal.omega0 …
  -/
  rw [range_omega, mem_setOf]
  /-
    🎉 no goals
  -/


/-- The "pre-aleph" function gives the cardinals listed by their ordinal index. `preAleph n = n`,
`preAleph ω = ℵ₀`, `preAleph (ω + 1) = succ ℵ₀`, etc.

For the more common aleph function skipping over finite cardinals, see `Cardinal.aleph`. -/
def preAleph : Ordinal.{u} ≃o Cardinal.{u} :=
  (enumOrdOrderIso _ not_bddAbove_isInitial).trans isInitialIso


@[simp]
theorem _root_.Ordinal.card_preOmega (o : Ordinal) : (preOmega o).card = preAleph o :=
  rfl


@[simp]
theorem ord_preAleph (o : Ordinal) : (preAleph o).ord = preOmega o := by
  /-
    o : Ordinal.{u_1}
    ⊢ Eq (Cardinal.preAleph o).ord (Ordinal.preOmega o)
  -/
  rw [← o.card_preOmega, (isInitial_preOmega o).ord_card]
  /-
    🎉 no goals
  -/


@[simp]
theorem type_cardinal : typeLT Cardinal = Ordinal.univ.{u, u + 1} := by
  /-
    ⊢ Eq (Ordinal.type fun x1 x2 => LT.lt x1 x2) Ordinal.univ.{u, u + 1}
  -/
  rw [Ordinal.univ_id]
  /-
    ⊢ Eq (Ordinal.type fun x1 x2 => LT.lt x1 x2) (Ordinal.type fun x1 x2 => LT.lt  …
  -/
  exact Quotient.sound ⟨preAleph.symm.toRelIsoLT⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem mk_cardinal : #Cardinal = univ.{u, u + 1} := by
  /-
    ⊢ Eq (Cardinal.mk Cardinal.{u}) Cardinal.univ.{u, u + 1}
  -/
  simpa only [card_type, card_univ] using congr_arg card type_cardinal
  /-
    🎉 no goals
  -/


theorem preAleph_lt_preAleph {o₁ o₂ : Ordinal} : preAleph o₁ < preAleph o₂ ↔ o₁ < o₂ :=
  preAleph.lt_iff_lt


theorem preAleph_le_preAleph {o₁ o₂ : Ordinal} : preAleph o₁ ≤ preAleph o₂ ↔ o₁ ≤ o₂ :=
  preAleph.le_iff_le


theorem preAleph_max (o₁ o₂ : Ordinal) : preAleph (max o₁ o₂) = max (preAleph o₁) (preAleph o₂) :=
  preAleph.monotone.map_max


@[simp]
theorem preAleph_zero : preAleph 0 = 0 :=
  preAleph.map_bot


@[simp]
theorem preAleph_succ (o : Ordinal) : preAleph (succ o) = succ (preAleph o) :=
  preAleph.map_succ o


@[simp]
theorem preAleph_nat (n : ℕ) : preAleph n = n := by
  /-
    n : Nat
    ⊢ Eq (Cardinal.preAleph ↑n) ↑n
  -/
  rw [← card_preOmega, preOmega_natCast, card_nat]
  /-
    🎉 no goals
  -/


@[simp]
theorem preAleph_omega0 : preAleph ω = ℵ₀ := by
  /-
    ⊢ Eq (Cardinal.preAleph Ordinal.omega0) Cardinal.aleph0
  -/
  rw [← card_preOmega, preOmega_omega0, card_omega0]
  /-
    🎉 no goals
  -/


@[simp]
theorem preAleph_pos {o : Ordinal} : 0 < preAleph o ↔ 0 < o := by
  /-
    o : Ordinal.{u_1}
    ⊢ Iff (LT.lt 0 (Cardinal.preAleph o)) (LT.lt 0 o)
  -/
  rw [← preAleph_zero, preAleph_lt_preAleph]
  /-
    🎉 no goals
  -/


@[simp]
theorem aleph0_le_preAleph {o : Ordinal} : ℵ₀ ≤ preAleph o ↔ ω ≤ o := by
  /-
    o : Ordinal.{u_1}
    ⊢ Iff (LE.le Cardinal.aleph0 (Cardinal.preAleph o)) (LE.le Ordinal.omega0 o)
  -/
  rw [← preAleph_omega0, preAleph_le_preAleph]
  /-
    🎉 no goals
  -/


@[simp]
theorem lift_preAleph (o : Ordinal.{u}) : lift.{v} (preAleph o) = preAleph (Ordinal.lift.{v} o) :=
  (preAleph.toInitialSeg.trans liftInitialSeg).eq
    (Ordinal.liftInitialSeg.trans preAleph.toInitialSeg) o


@[simp]
theorem _root_.Ordinal.lift_preOmega (o : Ordinal.{u}) :
    Ordinal.lift.{v} (preOmega o) = preOmega (Ordinal.lift.{v} o) := by
  /-
    o : Ordinal.{u}
    ⊢ Eq (Ordinal.lift.{v, u} (Ordinal.preOmega o)) (Ordinal.preOmega (Ordinal.lif …
  -/
  rw [← ord_preAleph, lift_ord, lift_preAleph, ord_preAleph]
  /-
    🎉 no goals
  -/


theorem preAleph_le_of_isLimit {o : Ordinal} (l : o.IsLimit) {c} :
    preAleph o ≤ c ↔ ∀ o' < o, preAleph o' ≤ c :=
  ⟨fun h o' h' => (preAleph_le_preAleph.2 <| h'.le).trans h, fun h => by
    /-
      o : Ordinal.{u_1}
      l : o.IsLimit
      c : Cardinal.{u_1}
      h : ∀ (o' : Ordinal.{u_1}), LT.lt o' o → LE.le (Cardinal.preAleph o') c
      ⊢ LE.le (Cardinal.preAleph o) c
    -/
    rw [← preAleph.apply_symm_apply c, preAleph_le_preAleph, limit_le l]
    /-
      o : Ordinal.{u_1}
      l : o.IsLimit
      c : Cardinal.{u_1}
      h : ∀ (o' : Ordinal.{u_1}), LT.lt o' o → LE.le (Cardinal.preAleph o') c
      ⊢ ∀ (x : Ordinal.{u_1}), LT.lt x o → LE.le x (Cardinal.preAleph.symm c)
    -/
    intro x h'
    /-
      o : Ordinal.{u_1}
      l : o.IsLimit
      c : Cardinal.{u_1}
      h : ∀ (o' : Ordinal.{u_1}), LT.lt o' o → LE.le (Cardinal.preAleph o') c
      x : Ordinal.{u_1}
      h' : LT.lt x o
      ⊢ LE.le x (Cardinal.preAleph.symm c)
    -/
    rw [← preAleph_le_preAleph, preAleph.apply_symm_apply]
    /-
      o : Ordinal.{u_1}
      l : o.IsLimit
      c : Cardinal.{u_1}
      h : ∀ (o' : Ordinal.{u_1}), LT.lt o' o → LE.le (Cardinal.preAleph o') c
      x : Ordinal.{u_1}
      h' : LT.lt x o
      ⊢ LE.le (Cardinal.preAleph x) c
    -/
    exact h _ h'⟩
    /-
      🎉 no goals
    -/


theorem preAleph_limit {o : Ordinal} (ho : o.IsLimit) : preAleph o = ⨆ a : Iio o, preAleph a := by
  /-
    o : Ordinal.{u_1}
    ho : o.IsLimit
    ⊢ Eq (Cardinal.preAleph o) (iSup fun a => Cardinal.preAleph ↑a)
  -/
  refine le_antisymm ?_ (ciSup_le' fun i => preAleph_le_preAleph.2 i.2.le)
  /-
    o : Ordinal.{u_1}
    ho : o.IsLimit
    ⊢ LE.le (Cardinal.preAleph o) (iSup fun a => Cardinal.preAleph ↑a)
  -/
  rw [preAleph_le_of_isLimit ho]
  /-
    o : Ordinal.{u_1}
    ho : o.IsLimit
    ⊢ ∀ (o' : Ordinal.{u_1}), LT.lt o' o → LE.le (Cardinal.preAleph o') (iSup fun  …
  -/
  exact fun a ha => le_ciSup (bddAbove_of_small _) (⟨a, ha⟩ : Iio o)
  /-
    🎉 no goals
  -/


/-- The `aleph` function gives the infinite cardinals listed by their ordinal index. `aleph 0 = ℵ₀`,
`aleph 1 = succ ℵ₀` is the first uncountable cardinal, and so on.

For a version including finite cardinals, see `Cardinal.aleph'`. -/
def aleph : Ordinal ↪o Cardinal :=
  (OrderEmbedding.addLeft ω).trans preAleph.toOrderEmbedding


@[inherit_doc]
scoped notation "ℵ_ " => aleph


/-- `ℵ₁` is the first uncountable cardinal. -/
scoped notation "ℵ₁" => ℵ_ 1


theorem aleph_eq_preAleph (o : Ordinal) : ℵ_ o = preAleph (ω + o) :=
  rfl


@[simp]
theorem _root_.Ordinal.card_omega (o : Ordinal) : (ω_ o).card = ℵ_ o :=
  rfl


@[simp]
theorem ord_aleph (o : Ordinal) : (ℵ_ o).ord = ω_ o :=
  ord_preAleph _


theorem aleph_lt_aleph {o₁ o₂ : Ordinal} : ℵ_ o₁ < ℵ_ o₂ ↔ o₁ < o₂ :=
  aleph.lt_iff_lt


@[deprecated aleph_lt_aleph (since := "2024-10-22")]
alias aleph_lt := aleph_lt_aleph


theorem aleph_le_aleph {o₁ o₂ : Ordinal} : ℵ_ o₁ ≤ ℵ_ o₂ ↔ o₁ ≤ o₂ :=
  aleph.le_iff_le


@[deprecated aleph_le_aleph (since := "2024-10-22")]
alias aleph_le := aleph_le_aleph


theorem aleph_max (o₁ o₂ : Ordinal) : ℵ_ (max o₁ o₂) = max (ℵ_ o₁) (ℵ_ o₂) :=
  aleph.monotone.map_max


@[deprecated aleph_max (since := "2024-08-28")]
theorem max_aleph_eq (o₁ o₂ : Ordinal) : max (ℵ_ o₁) (ℵ_ o₂) = ℵ_ (max o₁ o₂) :=
  (aleph_max o₁ o₂).symm


theorem preAleph_le_aleph (o : Ordinal) : preAleph o ≤ ℵ_ o :=
  preAleph_le_preAleph.2 (Ordinal.le_add_left _ _)


@[simp]
theorem aleph_succ (o : Ordinal) : ℵ_ (succ o) = succ (ℵ_ o) := by
  /-
    o : Ordinal.{u_1}
    ⊢ Eq (Cardinal.aleph (Order.succ o)) (Order.succ (Cardinal.aleph o))
  -/
  rw [aleph_eq_preAleph, add_succ, preAleph_succ, aleph_eq_preAleph]
  /-
    🎉 no goals
  -/


@[simp]
                                     /-
                                       ⊢ Eq (Cardinal.aleph 0) Cardinal.aleph0
                                     -/
theorem aleph_zero : ℵ_ 0 = ℵ₀ := by rw [aleph_eq_preAleph, add_zero, preAleph_omega0]
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem lift_aleph (o : Ordinal.{u}) : lift.{v} (aleph o) = aleph (Ordinal.lift.{v} o) := by
  /-
    o : Ordinal.{u}
    ⊢ Eq (Cardinal.lift.{v, u} (Cardinal.aleph o)) (Cardinal.aleph (Ordinal.lift.{ …
  -/
  simp [aleph_eq_preAleph]
  /-
    🎉 no goals
  -/


/-- For the theorem `lift ω = ω`, see `lift_omega0`. -/
@[simp]
theorem _root_.Ordinal.lift_omega (o : Ordinal.{u}) :
    Ordinal.lift.{v} (ω_ o) = ω_ (Ordinal.lift.{v} o) := by
  /-
    o : Ordinal.{u}
    ⊢ Eq (Ordinal.lift.{v, u} (Ordinal.omega o)) (Ordinal.omega (Ordinal.lift.{v,  …
  -/
  simp [omega_eq_preOmega]
  /-
    🎉 no goals
  -/


theorem aleph_limit {o : Ordinal} (ho : o.IsLimit) : ℵ_ o = ⨆ a : Iio o, ℵ_ a := by
  /-
    o : Ordinal.{u_1}
    ho : o.IsLimit
    ⊢ Eq (Cardinal.aleph o) (iSup fun a => Cardinal.aleph ↑a)
  -/
  rw [aleph_eq_preAleph, preAleph_limit (isLimit_add ω ho)]
  /-
    o : Ordinal.{u_1}
    ho : o.IsLimit
    ⊢ Eq (iSup fun a => Cardinal.preAleph ↑a) (iSup fun a => Cardinal.aleph ↑a)
  -/
  apply le_antisymm <;>
    /-
      case a
      o : Ordinal.{u_1}
      ho : o.IsLimit
      ⊢ LE.le (iSup fun a => Cardinal.preAleph ↑a) (iSup fun a => Cardinal.aleph ↑a)
    -/
    apply ciSup_mono' (bddAbove_of_small _) <;>
    /-
      case a
      o : Ordinal.{u_1}
      ho : o.IsLimit
      ⊢ ∀ (i : ↑(Set.Iio (HAdd.hAdd Ordinal.omega0 o))), Exists fun i' => LE.le (Car …
    -/
    intro i
    /-
      case a
      o : Ordinal.{u_1}
      ho : o.IsLimit
      i : ↑(Set.Iio (HAdd.hAdd Ordinal.omega0 o))
      ⊢ Exists fun i' => LE.le (Cardinal.preAleph ↑i) (Cardinal.aleph ↑i')
    -/
  · refine ⟨⟨_, sub_lt_of_lt_add i.2 ho.pos⟩, ?_⟩
    /-
      case a
      o : Ordinal.{u_1}
      ho : o.IsLimit
      i : ↑(Set.Iio (HAdd.hAdd Ordinal.omega0 o))
      ⊢ LE.le (Cardinal.preAleph ↑i) (Cardinal.aleph ↑⟨HSub.hSub (↑i) Ordinal.omega0 …
    -/
    simpa [aleph_eq_preAleph] using le_add_sub _ _
    /-
      🎉 no goals
    -/
    /-
      case a
      o : Ordinal.{u_1}
      ho : o.IsLimit
      i : ↑(Set.Iio o)
      ⊢ Exists fun i' => LE.le (Cardinal.aleph ↑i) (Cardinal.preAleph ↑i')
    -/
  · exact ⟨⟨_, add_lt_add_left i.2 ω⟩, le_rfl⟩
    /-
      🎉 no goals
    -/


theorem aleph0_le_aleph (o : Ordinal) : ℵ₀ ≤ ℵ_ o := by
  /-
    o : Ordinal.{u_1}
    ⊢ LE.le Cardinal.aleph0 (Cardinal.aleph o)
  -/
  rw [aleph_eq_preAleph, aleph0_le_preAleph]
  /-
    o : Ordinal.{u_1}
    ⊢ LE.le Ordinal.omega0 (HAdd.hAdd Ordinal.omega0 o)
  -/
  apply Ordinal.le_add_right
  /-
    🎉 no goals
  -/


theorem aleph_pos (o : Ordinal) : 0 < ℵ_ o :=
  aleph0_pos.trans_le (aleph0_le_aleph o)


@[simp]
theorem aleph_toNat (o : Ordinal) : toNat (ℵ_ o) = 0 :=
  toNat_apply_of_aleph0_le <| aleph0_le_aleph o


@[simp]
theorem aleph_toENat (o : Ordinal) : toENat (ℵ_ o) = ⊤ :=
  (toENat_eq_top.2 (aleph0_le_aleph o))


theorem isLimit_omega (o : Ordinal) : Ordinal.IsLimit (ω_ o) := by
  /-
    o : Ordinal.{u_1}
    ⊢ (Ordinal.omega o).IsLimit
  -/
  rw [← ord_aleph]
  /-
    o : Ordinal.{u_1}
    ⊢ (Cardinal.aleph o).ord.IsLimit
  -/
  exact isLimit_ord (aleph0_le_aleph _)
  /-
    🎉 no goals
  -/


@[deprecated isLimit_omega (since := "2024-10-24")]
theorem ord_aleph_isLimit (o : Ordinal) : (ℵ_ o).ord.IsLimit :=
  isLimit_ord <| aleph0_le_aleph _


@[simp]
theorem range_aleph : range aleph = Set.Ici ℵ₀ := by
  /-
    ⊢ Eq (Set.range ⇑Cardinal.aleph) (Set.Ici Cardinal.aleph0)
  -/
  ext c
  /-
    case h
    c : Cardinal.{u_1}
    ⊢ Iff (Membership.mem (Set.range ⇑Cardinal.aleph) c) (Membership.mem (Set.Ici  …
  -/
  refine ⟨fun ⟨o, e⟩ => e ▸ aleph0_le_aleph _, fun hc ↦ ⟨preAleph.symm c - ω, ?_⟩⟩
  /-
    case h
    c : Cardinal.{u_1}
    hc : Membership.mem (Set.Ici Cardinal.aleph0) c
    ⊢ Eq (Cardinal.aleph (HSub.hSub (Cardinal.preAleph.symm c) Ordinal.omega0)) c
  -/
  rw [aleph_eq_preAleph, Ordinal.add_sub_cancel_of_le, preAleph.apply_symm_apply]
  /-
    case h
    c : Cardinal.{u_1}
    hc : Membership.mem (Set.Ici Cardinal.aleph0) c
    ⊢ LE.le Ordinal.omega0 (Cardinal.preAleph.symm c)
  -/
  rwa [← aleph0_le_preAleph, preAleph.apply_symm_apply]
  /-
    🎉 no goals
  -/


theorem mem_range_aleph_iff {c : Cardinal} : c ∈ range aleph ↔ ℵ₀ ≤ c := by
  /-
    c : Cardinal.{u_1}
    ⊢ Iff (Membership.mem (Set.range ⇑Cardinal.aleph) c) (LE.le Cardinal.aleph0 c)
  -/
  rw [range_aleph, mem_Ici]
  /-
    🎉 no goals
  -/


@[deprecated mem_range_aleph_iff (since := "2024-10-24")]
theorem exists_aleph {c : Cardinal} : ℵ₀ ≤ c ↔ ∃ o, c = ℵ_ o :=
  ⟨fun h =>
    ⟨preAleph.symm c - ω, by
      /-
        c : Cardinal.{u_1}
        h : LE.le Cardinal.aleph0 c
        ⊢ Eq c (Cardinal.aleph (HSub.hSub (Cardinal.preAleph.symm c) Ordinal.omega0))
      -/
      rw [aleph_eq_preAleph, Ordinal.add_sub_cancel_of_le, preAleph.apply_symm_apply]
      /-
        c : Cardinal.{u_1}
        h : LE.le Cardinal.aleph0 c
        ⊢ LE.le Ordinal.omega0 (Cardinal.preAleph.symm c)
      -/
      rwa [← aleph0_le_preAleph, preAleph.apply_symm_apply]⟩,
      /-
        🎉 no goals
      -/
    fun ⟨o, e⟩ => e.symm ▸ aleph0_le_aleph _⟩


@[deprecated isNormal_preOmega (since := "2024-10-11")]
theorem preAleph_isNormal : IsNormal (ord ∘ preAleph) := by
  /-
    ⊢ Ordinal.IsNormal (Function.comp Cardinal.ord ⇑Cardinal.preAleph)
  -/
  convert isNormal_preOmega
  /-
    case h.e'_1
    ⊢ Eq (Function.comp Cardinal.ord ⇑Cardinal.preAleph) ⇑Ordinal.preOmega
  -/
  exact funext ord_preAleph
  /-
    🎉 no goals
  -/


@[deprecated isNormal_omega (since := "2024-10-11")]
theorem aleph_isNormal : IsNormal (ord ∘ aleph) := by
  /-
    ⊢ Ordinal.IsNormal (Function.comp Cardinal.ord ⇑Cardinal.aleph)
  -/
  convert isNormal_omega
  /-
    case h.e'_1
    ⊢ Eq (Function.comp Cardinal.ord ⇑Cardinal.aleph) ⇑Ordinal.omega
  -/
  exact funext ord_aleph
  /-
    🎉 no goals
  -/


@[simp]
theorem succ_aleph0 : succ ℵ₀ = ℵ₁ := by
  /-
    ⊢ Eq (Order.succ Cardinal.aleph0) (Cardinal.aleph 1)
  -/
  rw [← aleph_zero, ← aleph_succ, Ordinal.succ_zero]
  /-
    🎉 no goals
  -/


theorem aleph0_lt_aleph_one : ℵ₀ < ℵ₁ := by
  /-
    ⊢ LT.lt Cardinal.aleph0 (Cardinal.aleph 1)
  -/
  rw [← succ_aleph0]
  /-
    ⊢ LT.lt Cardinal.aleph0 (Order.succ Cardinal.aleph0)
  -/
  apply lt_succ
  /-
    🎉 no goals
  -/


theorem countable_iff_lt_aleph_one {α : Type*} (s : Set α) : s.Countable ↔ #s < ℵ₁ := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Iff s.Countable (LT.lt (Cardinal.mk ↑s) (Cardinal.aleph 1))
  -/
  rw [← succ_aleph0, lt_succ_iff, le_aleph0_iff_set_countable]
  /-
    🎉 no goals
  -/


@[simp]
theorem aleph1_le_lift {c : Cardinal.{u}} : ℵ₁ ≤ lift.{v} c ↔ ℵ₁ ≤ c := by
  /-
    c : Cardinal.{u}
    ⊢ Iff (LE.le (Cardinal.aleph 1) (Cardinal.lift.{v, u} c)) (LE.le (Cardinal.ale …
  -/
  simpa using lift_le (a := ℵ₁)
  /-
    🎉 no goals
  -/


@[simp]
theorem lift_le_aleph1 {c : Cardinal.{u}} : lift.{v} c ≤ ℵ₁ ↔ c ≤ ℵ₁ := by
  /-
    c : Cardinal.{u}
    ⊢ Iff (LE.le (Cardinal.lift.{v, u} c) (Cardinal.aleph 1)) (LE.le c (Cardinal.a …
  -/
  simpa using lift_le (b := ℵ₁)
  /-
    🎉 no goals
  -/


@[simp]
theorem aleph1_lt_lift {c : Cardinal.{u}} : ℵ₁ < lift.{v} c ↔ ℵ₁ < c := by
  /-
    c : Cardinal.{u}
    ⊢ Iff (LT.lt (Cardinal.aleph 1) (Cardinal.lift.{v, u} c)) (LT.lt (Cardinal.ale …
  -/
  simpa using lift_lt (a := ℵ₁)
  /-
    🎉 no goals
  -/


@[simp]
theorem lift_lt_aleph1 {c : Cardinal.{u}} : lift.{v} c < ℵ₁ ↔ c < ℵ₁ := by
  /-
    c : Cardinal.{u}
    ⊢ Iff (LT.lt (Cardinal.lift.{v, u} c) (Cardinal.aleph 1)) (LT.lt c (Cardinal.a …
  -/
  simpa using lift_lt (b := ℵ₁)
  /-
    🎉 no goals
  -/


@[simp]
theorem aleph1_eq_lift {c : Cardinal.{u}} : ℵ₁ = lift.{v} c ↔ ℵ₁ = c := by
  /-
    c : Cardinal.{u}
    ⊢ Iff (Eq (Cardinal.aleph 1) (Cardinal.lift.{v, u} c)) (Eq (Cardinal.aleph 1) c)
  -/
  simpa using lift_inj (a := ℵ₁)
  /-
    🎉 no goals
  -/


@[simp]
theorem lift_eq_aleph1 {c : Cardinal.{u}} : lift.{v} c = ℵ₁ ↔ c = ℵ₁ := by
  /-
    c : Cardinal.{u}
    ⊢ Iff (Eq (Cardinal.lift.{v, u} c) (Cardinal.aleph 1)) (Eq c (Cardinal.aleph 1))
  -/
  simpa using lift_inj (b := ℵ₁)
  /-
    🎉 no goals
  -/


theorem lt_omega_iff_card_lt {x o : Ordinal} : x < ω_ o ↔ x.card < ℵ_ o := by
  /-
    x o : Ordinal.{u_1}
    ⊢ Iff (LT.lt x (Ordinal.omega o)) (LT.lt x.card (Cardinal.aleph o))
  -/
  rw [← (isInitial_omega o).card_lt_card, card_omega]
  /-
    🎉 no goals
  -/


@[deprecated preAleph (since := "2024-10-22")]
noncomputable alias aleph' := preAleph


/-- The `aleph'` index function, which gives the ordinal index of a cardinal.
  (The `aleph'` part is because unlike `aleph` this counts also the
  finite stages. So `alephIdx n = n`, `alephIdx ω = ω`,
  `alephIdx ℵ₁ = ω + 1` and so on.)
  In this definition, we register additionally that this function is an initial segment,
  i.e., it is order preserving and its range is an initial segment of the ordinals.
  For the basic function version, see `alephIdx`.
  For an upgraded version stating that the range is everything, see `AlephIdx.rel_iso`. -/
@[deprecated preAleph (since := "2024-08-28")]
def alephIdx.initialSeg : @InitialSeg Cardinal Ordinal (· < ·) (· < ·) :=
  @RelEmbedding.collapse Cardinal Ordinal (· < ·) (· < ·) _ Cardinal.ord.orderEmbedding.ltEmbedding


set_option linter.deprecated false in
/-- The `aleph'` index function, which gives the ordinal index of a cardinal.
  (The `aleph'` part is because unlike `aleph` this counts also the
  finite stages. So `alephIdx n = n`, `alephIdx ℵ₀ = ω`,
  `alephIdx ℵ₁ = ω + 1` and so on.)
  In this version, we register additionally that this function is an order isomorphism
  between cardinals and ordinals.
  For the basic function version, see `alephIdx`. -/
@[deprecated preAleph (since := "2024-08-28")]
def alephIdx.relIso : @RelIso Cardinal.{u} Ordinal.{u} (· < ·) (· < ·) :=
  aleph'.symm.toRelIsoLT


set_option linter.deprecated false in
/-- The `aleph'` index function, which gives the ordinal index of a cardinal.
  (The `aleph'` part is because unlike `aleph` this counts also the
  finite stages. So `alephIdx n = n`, `alephIdx ω = ω`,
  `alephIdx ℵ₁ = ω + 1` and so on.)
  For an upgraded version stating that the range is everything, see `AlephIdx.rel_iso`. -/
@[deprecated aleph' (since := "2024-08-28")]
def alephIdx : Cardinal → Ordinal :=
  aleph'.symm


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-08-28")]
theorem alephIdx.relIso_coe : (alephIdx.relIso : Cardinal → Ordinal) = alephIdx :=
  rfl


set_option linter.deprecated false in
/-- The `aleph'` function gives the cardinals listed by their ordinal
  index, and is the inverse of `aleph_idx`.
  `aleph' n = n`, `aleph' ω = ω`, `aleph' (ω + 1) = succ ℵ₀`, etc.
  In this version, we register additionally that this function is an order isomorphism
  between ordinals and cardinals.
  For the basic function version, see `aleph'`. -/
@[deprecated aleph' (since := "2024-08-28")]
def Aleph'.relIso :=
  aleph'


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-08-28")]
theorem aleph'.relIso_coe : (Aleph'.relIso : Ordinal → Cardinal) = aleph' :=
  rfl


set_option linter.deprecated false in
@[deprecated preAleph_lt_preAleph (since := "2024-10-22")]
theorem aleph'_lt {o₁ o₂ : Ordinal} : aleph' o₁ < aleph' o₂ ↔ o₁ < o₂ :=
  aleph'.lt_iff_lt


set_option linter.deprecated false in
@[deprecated preAleph_le_preAleph (since := "2024-10-22")]
theorem aleph'_le {o₁ o₂ : Ordinal} : aleph' o₁ ≤ aleph' o₂ ↔ o₁ ≤ o₂ :=
  aleph'.le_iff_le


set_option linter.deprecated false in
@[deprecated preAleph_max (since := "2024-10-22")]
theorem aleph'_max (o₁ o₂ : Ordinal) : aleph' (max o₁ o₂) = max (aleph' o₁) (aleph' o₂) :=
  aleph'.monotone.map_max


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-08-28")]
theorem aleph'_alephIdx (c : Cardinal) : aleph' c.alephIdx = c :=
  Cardinal.alephIdx.relIso.toEquiv.symm_apply_apply c


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-08-28")]
theorem alephIdx_aleph' (o : Ordinal) : (aleph' o).alephIdx = o :=
  Cardinal.alephIdx.relIso.toEquiv.apply_symm_apply o


set_option linter.deprecated false in
@[deprecated preAleph_zero (since := "2024-10-22")]
theorem aleph'_zero : aleph' 0 = 0 :=
  aleph'.map_bot


set_option linter.deprecated false in
@[deprecated preAleph_succ (since := "2024-10-22")]
theorem aleph'_succ (o : Ordinal) : aleph' (succ o) = succ (aleph' o) :=
  aleph'.map_succ o


set_option linter.deprecated false in
@[deprecated preAleph_nat (since := "2024-10-22")]
theorem aleph'_nat : ∀ n : ℕ, aleph' n = n :=
  preAleph_nat


set_option linter.deprecated false in
@[deprecated lift_preAleph (since := "2024-10-22")]
theorem lift_aleph' (o : Ordinal.{u}) : lift.{v} (aleph' o) = aleph' (Ordinal.lift.{v} o) :=
  lift_preAleph o


set_option linter.deprecated false in
@[deprecated preAleph_le_of_isLimit (since := "2024-10-22")]
theorem aleph'_le_of_limit {o : Ordinal} (l : o.IsLimit) {c} :
    aleph' o ≤ c ↔ ∀ o' < o, aleph' o' ≤ c :=
  preAleph_le_of_isLimit l


set_option linter.deprecated false in
@[deprecated preAleph_limit (since := "2024-10-22")]
theorem aleph'_limit {o : Ordinal} (ho : o.IsLimit) : aleph' o = ⨆ a : Iio o, aleph' a :=
  preAleph_limit ho


set_option linter.deprecated false in
@[deprecated preAleph_omega0 (since := "2024-10-22")]
theorem aleph'_omega0 : aleph' ω = ℵ₀ :=
  preAleph_omega0


@[deprecated "No deprecation message was provided."  (since := "2024-09-30")]
alias aleph'_omega := aleph'_omega0


set_option linter.deprecated false in
/-- `aleph'` and `aleph_idx` form an equivalence between `Ordinal` and `Cardinal` -/
@[deprecated aleph' (since := "2024-08-28")]
def aleph'Equiv : Ordinal ≃ Cardinal :=
  ⟨aleph', alephIdx, alephIdx_aleph', aleph'_alephIdx⟩


@[deprecated aleph_eq_preAleph (since := "2024-10-22")]
theorem aleph_eq_aleph' (o : Ordinal) : ℵ_ o = preAleph (ω + o) :=
  rfl


set_option linter.deprecated false in
@[deprecated aleph0_le_preAleph (since := "2024-10-22")]
theorem aleph0_le_aleph' {o : Ordinal} : ℵ₀ ≤ aleph' o ↔ ω ≤ o := by
  /-
    o : Ordinal.{u_1}
    ⊢ Iff (LE.le Cardinal.aleph0 (Cardinal.aleph' o)) (LE.le Ordinal.omega0 o)
  -/
  rw [← aleph'_omega0, aleph'_le]
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated preAleph_pos (since := "2024-10-22")]
theorem aleph'_pos {o : Ordinal} (ho : 0 < o) : 0 < aleph' o := by
  /-
    o : Ordinal.{u_1}
    ho : LT.lt 0 o
    ⊢ LT.lt 0 (Cardinal.aleph' o)
  -/
  rwa [← aleph'_zero, aleph'_lt]
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated preAleph_isNormal (since := "2024-10-22")]
theorem aleph'_isNormal : IsNormal (ord ∘ aleph') :=
  preAleph_isNormal

-- TODO: these lemmas should be stated in terms of the `ω` function and of an `IsInitial` predicate,
-- neither of which currently exist.
--
-- They should also use `¬ BddAbove` instead of `Unbounded (· < ·)`.


/-- Ordinals that are cardinals are unbounded. -/
@[deprecated "No deprecation message was provided."  (since := "2024-09-24")]
theorem ord_card_unbounded : Unbounded (· < ·) { b : Ordinal | b.card.ord = b } :=
  unbounded_lt_iff.2 fun a =>
    ⟨_,
      ⟨by
        /-
          a : Ordinal.{u_1}
          ⊢ Membership.mem (setOf fun b => Eq b.card.ord b) (Order.succ a.card).ord
        -/
        dsimp
        /-
          a : Ordinal.{u_1}
          ⊢ Eq (Order.succ a.card).ord.card.ord (Order.succ a.card).ord
        -/
        rw [card_ord], (lt_ord_succ_card a).le⟩⟩
        /-
          🎉 no goals
        -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-09-24")]
theorem eq_aleph'_of_eq_card_ord {o : Ordinal} (ho : o.card.ord = o) : ∃ a, (aleph' a).ord = o :=
                          /-
                            o : Ordinal.{u_1}
                            ho : Eq o.card.ord o
                            ⊢ Eq (Cardinal.aleph' (Cardinal.aleph'.symm o.card)).ord o
                          -/
  ⟨aleph'.symm o.card, by simpa using ho⟩
                          /-
                            🎉 no goals
                          -/


set_option linter.deprecated false in
/-- Infinite ordinals that are cardinals are unbounded. -/
@[deprecated "No deprecation message was provided."  (since := "2024-09-24")]
theorem ord_card_unbounded' : Unbounded (· < ·) { b : Ordinal | b.card.ord = b ∧ ω ≤ b } :=
  (unbounded_lt_inter_le ω).2 ord_card_unbounded


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-09-24")]
theorem eq_aleph_of_eq_card_ord {o : Ordinal} (ho : o.card.ord = o) (ho' : ω ≤ o) :
    ∃ a, (ℵ_ a).ord = o := by
  /-
    o : Ordinal.{u_1}
    ho : Eq o.card.ord o
    ho' : LE.le Ordinal.omega0 o
    ⊢ Exists fun a => Eq (Cardinal.aleph a).ord o
  -/
  cases' eq_aleph'_of_eq_card_ord ho with a ha
  /-
    case intro
    o : Ordinal.{u_1}
    ho : Eq o.card.ord o
    ho' : LE.le Ordinal.omega0 o
    a : Ordinal.{u_1}
    ha : Eq (Cardinal.aleph' a).ord o
    ⊢ Exists fun a => Eq (Cardinal.aleph a).ord o
  -/
  use a - ω
  /-
    case h
    o : Ordinal.{u_1}
    ho : Eq o.card.ord o
    ho' : LE.le Ordinal.omega0 o
    a : Ordinal.{u_1}
    ha : Eq (Cardinal.aleph' a).ord o
    ⊢ Eq (Cardinal.aleph (HSub.hSub a Ordinal.omega0)).ord o
  -/
  rwa [aleph_eq_aleph', Ordinal.add_sub_cancel_of_le]
  /-
    case h
    o : Ordinal.{u_1}
    ho : Eq o.card.ord o
    ho' : LE.le Ordinal.omega0 o
    a : Ordinal.{u_1}
    ha : Eq (Cardinal.aleph' a).ord o
    ⊢ LE.le Ordinal.omega0 a
  -/
  rwa [← aleph0_le_aleph', ← ord_le_ord, ha, ord_aleph0]
  /-
    🎉 no goals
  -/


/-- Beth numbers are defined so that `beth 0 = ℵ₀`, `beth (succ o) = 2 ^ beth o`, and when `o` is
a limit ordinal, `beth o` is the supremum of `beth o'` for `o' < o`.

Assuming the generalized continuum hypothesis, which is undecidable in ZFC, `beth o = aleph o` for
every `o`. -/
def beth (o : Ordinal.{u}) : Cardinal.{u} :=
  limitRecOn o ℵ₀ (fun _ x => 2 ^ x) fun a _ IH => ⨆ b : Iio a, IH b.1 b.2


@[inherit_doc]
scoped notation "ℶ_ " => beth


@[simp]
theorem beth_zero : ℶ_ 0 = ℵ₀ :=
  limitRecOn_zero _ _ _


@[simp]
theorem beth_succ (o : Ordinal) : ℶ_ (succ o) = 2 ^ beth o :=
  limitRecOn_succ _ _ _ _


theorem beth_limit {o : Ordinal} : o.IsLimit → ℶ_ o = ⨆ a : Iio o, ℶ_ a :=
  limitRecOn_limit _ _ _ _


theorem beth_strictMono : StrictMono beth := by
  /-
    ⊢ StrictMono Cardinal.beth
  -/
  intro a b
  /-
    a b : Ordinal.{u_1}
    ⊢ LT.lt a b → LT.lt (Cardinal.beth a) (Cardinal.beth b)
  -/
  induction' b using Ordinal.induction with b IH generalizing a
  /-
    case h
    b : Ordinal.{u_1}
    IH : ∀ (k : Ordinal.{u_1}), LT.lt k b → ∀ ⦃a : Ordinal.{u_1}⦄, LT.lt a k → LT. …
    a : Ordinal.{u_1}
    ⊢ LT.lt a b → LT.lt (Cardinal.beth a) (Cardinal.beth b)
  -/
  intro h
  /-
    case h
    b : Ordinal.{u_1}
    IH : ∀ (k : Ordinal.{u_1}), LT.lt k b → ∀ ⦃a : Ordinal.{u_1}⦄, LT.lt a k → LT. …
    a : Ordinal.{u_1}
    h : LT.lt a b
    ⊢ LT.lt (Cardinal.beth a) (Cardinal.beth b)
  -/
  rcases zero_or_succ_or_limit b with (rfl | ⟨c, rfl⟩ | hb)
    /-
      case h.inl
      a : Ordinal.{u_1}
      IH : ∀ (k : Ordinal.{u_1}), LT.lt k 0 → ∀ ⦃a : Ordinal.{u_1}⦄, LT.lt a k → LT. …
      h : LT.lt a 0
      ⊢ LT.lt (Cardinal.beth a) (Cardinal.beth 0)
    -/
  · exact (Ordinal.not_lt_zero a h).elim
    /-
      🎉 no goals
    -/
    /-
      case h.inr.inl.intro
      a c : Ordinal.{u_1}
      IH : ∀ (k : Ordinal.{u_1}), LT.lt k (Order.succ c) → ∀ ⦃a : Ordinal.{u_1}⦄, LT …
      h : LT.lt a (Order.succ c)
      ⊢ LT.lt (Cardinal.beth a) (Cardinal.beth (Order.succ c))
    -/
  · rw [lt_succ_iff] at h
    /-
      case h.inr.inl.intro
      a c : Ordinal.{u_1}
      IH : ∀ (k : Ordinal.{u_1}), LT.lt k (Order.succ c) → ∀ ⦃a : Ordinal.{u_1}⦄, LT …
      h : LE.le a c
      ⊢ LT.lt (Cardinal.beth a) (Cardinal.beth (Order.succ c))
    -/
    rw [beth_succ]
    /-
      case h.inr.inl.intro
      a c : Ordinal.{u_1}
      IH : ∀ (k : Ordinal.{u_1}), LT.lt k (Order.succ c) → ∀ ⦃a : Ordinal.{u_1}⦄, LT …
      h : LE.le a c
      ⊢ LT.lt (Cardinal.beth a) (HPow.hPow 2 (Cardinal.beth c))
    -/
    apply lt_of_le_of_lt _ (cantor _)
    /-
      a c : Ordinal.{u_1}
      IH : ∀ (k : Ordinal.{u_1}), LT.lt k (Order.succ c) → ∀ ⦃a : Ordinal.{u_1}⦄, LT …
      h : LE.le a c
      ⊢ LE.le (Cardinal.beth a) (Cardinal.beth c)
    -/
    rcases eq_or_lt_of_le h with (rfl | h)
      /-
        case inl
        a : Ordinal.{u_1}
        IH : ∀ (k : Ordinal.{u_1}), LT.lt k (Order.succ a) → ∀ ⦃a : Ordinal.{u_1}⦄, LT …
        h : LE.le a a
        ⊢ LE.le (Cardinal.beth a) (Cardinal.beth a)
      -/
    · rfl
      /-
        🎉 no goals
      -/
    /-
      case inr
      a c : Ordinal.{u_1}
      IH : ∀ (k : Ordinal.{u_1}), LT.lt k (Order.succ c) → ∀ ⦃a : Ordinal.{u_1}⦄, LT …
      h✝ : LE.le a c
      h : LT.lt a c
      ⊢ LE.le (Cardinal.beth a) (Cardinal.beth c)
    -/
    exact (IH c (lt_succ c) h).le
    /-
      🎉 no goals
    -/
    /-
      case h.inr.inr
      b : Ordinal.{u_1}
      IH : ∀ (k : Ordinal.{u_1}), LT.lt k b → ∀ ⦃a : Ordinal.{u_1}⦄, LT.lt a k → LT. …
      a : Ordinal.{u_1}
      h : LT.lt a b
      hb : b.IsLimit
      ⊢ LT.lt (Cardinal.beth a) (Cardinal.beth b)
    -/
  · apply (cantor _).trans_le
    /-
      case h.inr.inr
      b : Ordinal.{u_1}
      IH : ∀ (k : Ordinal.{u_1}), LT.lt k b → ∀ ⦃a : Ordinal.{u_1}⦄, LT.lt a k → LT. …
      a : Ordinal.{u_1}
      h : LT.lt a b
      hb : b.IsLimit
      ⊢ LE.le (HPow.hPow 2 (Cardinal.beth a)) (Cardinal.beth b)
    -/
    rw [beth_limit hb, ← beth_succ]
    /-
      case h.inr.inr
      b : Ordinal.{u_1}
      IH : ∀ (k : Ordinal.{u_1}), LT.lt k b → ∀ ⦃a : Ordinal.{u_1}⦄, LT.lt a k → LT. …
      a : Ordinal.{u_1}
      h : LT.lt a b
      hb : b.IsLimit
      ⊢ LE.le (Cardinal.beth (Order.succ a)) (iSup fun a => Cardinal.beth ↑a)
    -/
    exact le_ciSup (bddAbove_of_small _) (⟨_, hb.succ_lt h⟩ : Iio b)
    /-
      🎉 no goals
    -/


theorem beth_mono : Monotone beth :=
  beth_strictMono.monotone


@[simp]
theorem beth_lt {o₁ o₂ : Ordinal} : ℶ_ o₁ < ℶ_ o₂ ↔ o₁ < o₂ :=
  beth_strictMono.lt_iff_lt


@[simp]
theorem beth_le {o₁ o₂ : Ordinal} : ℶ_ o₁ ≤ ℶ_ o₂ ↔ o₁ ≤ o₂ :=
  beth_strictMono.le_iff_le


theorem aleph_le_beth (o : Ordinal) : ℵ_ o ≤ ℶ_ o := by
  induction o using limitRecOn with
  | H₁ => simp
  | H₂ o h =>
    rw [aleph_succ, beth_succ, succ_le_iff]
    exact (cantor _).trans_le (power_le_power_left two_ne_zero h)
  | H₃ o ho IH =>
    rw [aleph_limit ho, beth_limit ho]
    exact ciSup_mono (bddAbove_of_small _) fun x => IH x.1 x.2


theorem aleph0_le_beth (o : Ordinal) : ℵ₀ ≤ ℶ_ o :=
  (aleph0_le_aleph o).trans <| aleph_le_beth o


theorem beth_pos (o : Ordinal) : 0 < ℶ_ o :=
  aleph0_pos.trans_le <| aleph0_le_beth o


theorem beth_ne_zero (o : Ordinal) : ℶ_ o ≠ 0 :=
  (beth_pos o).ne'


theorem isNormal_beth : IsNormal (ord ∘ beth) := by
  refine (isNormal_iff_strictMono_limit _).2
    ⟨ord_strictMono.comp beth_strictMono, fun o ho a ha ↦ ?_⟩
  /-
    o : Ordinal.{u_1}
    ho : o.IsLimit
    a : Ordinal.{u_1}
    ha : ∀ (b : Ordinal.{u_1}), LT.lt b o → LE.le (Function.comp Cardinal.ord Card …
    ⊢ LE.le (Function.comp Cardinal.ord Cardinal.beth o) a
  -/
  rw [comp_apply, beth_limit ho, ord_le]
  /-
    o : Ordinal.{u_1}
    ho : o.IsLimit
    a : Ordinal.{u_1}
    ha : ∀ (b : Ordinal.{u_1}), LT.lt b o → LE.le (Function.comp Cardinal.ord Card …
    ⊢ LE.le (iSup fun a => Cardinal.beth ↑a) a.card
  -/
  exact ciSup_le' fun b => ord_le.1 (ha _ b.2)
  /-
    🎉 no goals
  -/


@[deprecated isNormal_beth (since := "2024-10-11")]
theorem beth_normal : IsNormal.{u} fun o => (beth o).ord :=
  isNormal_beth


