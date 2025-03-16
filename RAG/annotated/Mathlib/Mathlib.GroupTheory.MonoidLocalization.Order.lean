@[to_additive]
instance le : LE (Localization s) :=
  ⟨fun a b =>
    Localization.liftOn₂ a b (fun a₁ a₂ b₁ b₂ => ↑b₂ * a₁ ≤ a₂ * b₁)
      fun {a₁ b₁ a₂ b₂ c₁ d₁ c₂ d₂} hab hcd => propext <| by
        /-
          α : Type u_1
          inst✝ : OrderedCancelCommMonoid α
          s : Submonoid α
          a₁✝ b₁✝ : α
          a₂✝ b₂✝ : Subtype fun x => Membership.mem s x
          a b : Localization s
          a₁ b₁ : α
          a₂ b₂ : Subtype fun x => Membership.mem s x
          c₁ d₁ : α
          c₂ d₂ : Subtype fun x => Membership.mem s x
          hab : (Localization.r s) { fst := a₁, snd := a₂ } { fst := b₁, snd := b₂ }
          hcd : (Localization.r s) { fst := c₁, snd := c₂ } { fst := d₁, snd := d₂ }
          ⊢ Iff ((fun a₁ a₂ b₁ b₂ => LE.le (HMul.hMul (↑b₂) a₁) (HMul.hMul (↑a₂) b₁)) a₁ …
        -/
        obtain ⟨e, he⟩ := r_iff_exists.1 hab
        /-
          case intro
          α : Type u_1
          inst✝ : OrderedCancelCommMonoid α
          s : Submonoid α
          a₁✝ b₁✝ : α
          a₂✝ b₂✝ : Subtype fun x => Membership.mem s x
          a b : Localization s
          a₁ b₁ : α
          a₂ b₂ : Subtype fun x => Membership.mem s x
          c₁ d₁ : α
          c₂ d₂ : Subtype fun x => Membership.mem s x
          hab : (Localization.r s) { fst := a₁, snd := a₂ } { fst := b₁, snd := b₂ }
          hcd : (Localization.r s) { fst := c₁, snd := c₂ } { fst := d₁, snd := d₂ }
          e : Subtype fun x => Membership.mem s x
          he : Eq (HMul.hMul (↑e) (HMul.hMul ↑{ fst := b₁, snd := b₂ }.2 { fst := a₁, sn …
          ⊢ Iff ((fun a₁ a₂ b₁ b₂ => LE.le (HMul.hMul (↑b₂) a₁) (HMul.hMul (↑a₂) b₁)) a₁ …
        -/
        obtain ⟨f, hf⟩ := r_iff_exists.1 hcd
        /-
          case intro.intro
          α : Type u_1
          inst✝ : OrderedCancelCommMonoid α
          s : Submonoid α
          a₁✝ b₁✝ : α
          a₂✝ b₂✝ : Subtype fun x => Membership.mem s x
          a b : Localization s
          a₁ b₁ : α
          a₂ b₂ : Subtype fun x => Membership.mem s x
          c₁ d₁ : α
          c₂ d₂ : Subtype fun x => Membership.mem s x
          hab : (Localization.r s) { fst := a₁, snd := a₂ } { fst := b₁, snd := b₂ }
          hcd : (Localization.r s) { fst := c₁, snd := c₂ } { fst := d₁, snd := d₂ }
          e : Subtype fun x => Membership.mem s x
          he : Eq (HMul.hMul (↑e) (HMul.hMul ↑{ fst := b₁, snd := b₂ }.2 { fst := a₁, sn …
          f : Subtype fun x => Membership.mem s x
          hf : Eq (HMul.hMul (↑f) (HMul.hMul ↑{ fst := d₁, snd := d₂ }.2 { fst := c₁, sn …
          ⊢ Iff ((fun a₁ a₂ b₁ b₂ => LE.le (HMul.hMul (↑b₂) a₁) (HMul.hMul (↑a₂) b₁)) a₁ …
        -/
        simp only [mul_right_inj] at he hf
        /-
          case intro.intro
          α : Type u_1
          inst✝ : OrderedCancelCommMonoid α
          s : Submonoid α
          a₁✝ b₁✝ : α
          a₂✝ b₂✝ : Subtype fun x => Membership.mem s x
          a b : Localization s
          a₁ b₁ : α
          a₂ b₂ : Subtype fun x => Membership.mem s x
          c₁ d₁ : α
          c₂ d₂ : Subtype fun x => Membership.mem s x
          hab : (Localization.r s) { fst := a₁, snd := a₂ } { fst := b₁, snd := b₂ }
          hcd : (Localization.r s) { fst := c₁, snd := c₂ } { fst := d₁, snd := d₂ }
          e f : Subtype fun x => Membership.mem s x
          he : Eq (HMul.hMul (↑b₂) a₁) (HMul.hMul (↑a₂) b₁)
          hf : Eq (HMul.hMul (↑d₂) c₁) (HMul.hMul (↑c₂) d₁)
          ⊢ Iff ((fun a₁ a₂ b₁ b₂ => LE.le (HMul.hMul (↑b₂) a₁) (HMul.hMul (↑a₂) b₁)) a₁ …
        -/
        dsimp
        rw [← mul_le_mul_iff_right, mul_right_comm, ← hf, mul_right_comm, mul_right_comm (a₂ : α),
          mul_le_mul_iff_right, ← mul_le_mul_iff_left, mul_left_comm, he, mul_left_comm,
          mul_left_comm (b₂ : α), mul_le_mul_iff_left]⟩


@[to_additive]
instance lt : LT (Localization s) :=
  ⟨fun a b =>
    Localization.liftOn₂ a b (fun a₁ a₂ b₁ b₂ => ↑b₂ * a₁ < a₂ * b₁)
      fun {a₁ b₁ a₂ b₂ c₁ d₁ c₂ d₂} hab hcd => propext <| by
        /-
          α : Type u_1
          inst✝ : OrderedCancelCommMonoid α
          s : Submonoid α
          a₁✝ b₁✝ : α
          a₂✝ b₂✝ : Subtype fun x => Membership.mem s x
          a b : Localization s
          a₁ b₁ : α
          a₂ b₂ : Subtype fun x => Membership.mem s x
          c₁ d₁ : α
          c₂ d₂ : Subtype fun x => Membership.mem s x
          hab : (Localization.r s) { fst := a₁, snd := a₂ } { fst := b₁, snd := b₂ }
          hcd : (Localization.r s) { fst := c₁, snd := c₂ } { fst := d₁, snd := d₂ }
          ⊢ Iff ((fun a₁ a₂ b₁ b₂ => LT.lt (HMul.hMul (↑b₂) a₁) (HMul.hMul (↑a₂) b₁)) a₁ …
        -/
        obtain ⟨e, he⟩ := r_iff_exists.1 hab
        /-
          case intro
          α : Type u_1
          inst✝ : OrderedCancelCommMonoid α
          s : Submonoid α
          a₁✝ b₁✝ : α
          a₂✝ b₂✝ : Subtype fun x => Membership.mem s x
          a b : Localization s
          a₁ b₁ : α
          a₂ b₂ : Subtype fun x => Membership.mem s x
          c₁ d₁ : α
          c₂ d₂ : Subtype fun x => Membership.mem s x
          hab : (Localization.r s) { fst := a₁, snd := a₂ } { fst := b₁, snd := b₂ }
          hcd : (Localization.r s) { fst := c₁, snd := c₂ } { fst := d₁, snd := d₂ }
          e : Subtype fun x => Membership.mem s x
          he : Eq (HMul.hMul (↑e) (HMul.hMul ↑{ fst := b₁, snd := b₂ }.2 { fst := a₁, sn …
          ⊢ Iff ((fun a₁ a₂ b₁ b₂ => LT.lt (HMul.hMul (↑b₂) a₁) (HMul.hMul (↑a₂) b₁)) a₁ …
        -/
        obtain ⟨f, hf⟩ := r_iff_exists.1 hcd
        /-
          case intro.intro
          α : Type u_1
          inst✝ : OrderedCancelCommMonoid α
          s : Submonoid α
          a₁✝ b₁✝ : α
          a₂✝ b₂✝ : Subtype fun x => Membership.mem s x
          a b : Localization s
          a₁ b₁ : α
          a₂ b₂ : Subtype fun x => Membership.mem s x
          c₁ d₁ : α
          c₂ d₂ : Subtype fun x => Membership.mem s x
          hab : (Localization.r s) { fst := a₁, snd := a₂ } { fst := b₁, snd := b₂ }
          hcd : (Localization.r s) { fst := c₁, snd := c₂ } { fst := d₁, snd := d₂ }
          e : Subtype fun x => Membership.mem s x
          he : Eq (HMul.hMul (↑e) (HMul.hMul ↑{ fst := b₁, snd := b₂ }.2 { fst := a₁, sn …
          f : Subtype fun x => Membership.mem s x
          hf : Eq (HMul.hMul (↑f) (HMul.hMul ↑{ fst := d₁, snd := d₂ }.2 { fst := c₁, sn …
          ⊢ Iff ((fun a₁ a₂ b₁ b₂ => LT.lt (HMul.hMul (↑b₂) a₁) (HMul.hMul (↑a₂) b₁)) a₁ …
        -/
        simp only [mul_right_inj] at he hf
        /-
          case intro.intro
          α : Type u_1
          inst✝ : OrderedCancelCommMonoid α
          s : Submonoid α
          a₁✝ b₁✝ : α
          a₂✝ b₂✝ : Subtype fun x => Membership.mem s x
          a b : Localization s
          a₁ b₁ : α
          a₂ b₂ : Subtype fun x => Membership.mem s x
          c₁ d₁ : α
          c₂ d₂ : Subtype fun x => Membership.mem s x
          hab : (Localization.r s) { fst := a₁, snd := a₂ } { fst := b₁, snd := b₂ }
          hcd : (Localization.r s) { fst := c₁, snd := c₂ } { fst := d₁, snd := d₂ }
          e f : Subtype fun x => Membership.mem s x
          he : Eq (HMul.hMul (↑b₂) a₁) (HMul.hMul (↑a₂) b₁)
          hf : Eq (HMul.hMul (↑d₂) c₁) (HMul.hMul (↑c₂) d₁)
          ⊢ Iff ((fun a₁ a₂ b₁ b₂ => LT.lt (HMul.hMul (↑b₂) a₁) (HMul.hMul (↑a₂) b₁)) a₁ …
        -/
        dsimp
        rw [← mul_lt_mul_iff_right, mul_right_comm, ← hf, mul_right_comm, mul_right_comm (a₂ : α),
          mul_lt_mul_iff_right, ← mul_lt_mul_iff_left, mul_left_comm, he, mul_left_comm,
          mul_left_comm (b₂ : α), mul_lt_mul_iff_left]⟩


@[to_additive]
theorem mk_le_mk : mk a₁ a₂ ≤ mk b₁ b₂ ↔ ↑b₂ * a₁ ≤ a₂ * b₁ :=
  Iff.rfl


@[to_additive]
theorem mk_lt_mk : mk a₁ a₂ < mk b₁ b₂ ↔ ↑b₂ * a₁ < a₂ * b₁ :=
  Iff.rfl

-- declaring this separately to the instance below makes things faster

@[to_additive]
instance partialOrder : PartialOrder (Localization s) where
  le := (· ≤ ·)
  lt := (· < ·)
  le_refl a := Localization.induction_on a fun _ => le_rfl
  le_trans a b c :=
    Localization.induction_on₃ a b c fun a b c hab hbc => by
      /-
        α : Type u_1
        inst✝ : OrderedCancelCommMonoid α
        s : Submonoid α
        a₁ b₁ : α
        a₂ b₂ : Subtype fun x => Membership.mem s x
        a✝ b✝ c✝ : Localization s
        a b c : Prod α (Subtype fun x => Membership.mem s x)
        hab : LE.le (Localization.mk a.1 a.2) (Localization.mk b.1 b.2)
        hbc : LE.le (Localization.mk b.1 b.2) (Localization.mk c.1 c.2)
        ⊢ LE.le (Localization.mk a.1 a.2) (Localization.mk c.1 c.2)
      -/
      simp only [mk_le_mk] at hab hbc ⊢
      /-
        α : Type u_1
        inst✝ : OrderedCancelCommMonoid α
        s : Submonoid α
        a₁ b₁ : α
        a₂ b₂ : Subtype fun x => Membership.mem s x
        a✝ b✝ c✝ : Localization s
        a b c : Prod α (Subtype fun x => Membership.mem s x)
        hab : LE.le (HMul.hMul (↑b.2) a.1) (HMul.hMul (↑a.2) b.1)
        hbc : LE.le (HMul.hMul (↑c.2) b.1) (HMul.hMul (↑b.2) c.1)
        ⊢ LE.le (HMul.hMul (↑c.2) a.1) (HMul.hMul (↑a.2) c.1)
      -/
      apply le_of_mul_le_mul_left' _
        /-
          α : Type u_1
          inst✝ : OrderedCancelCommMonoid α
          s : Submonoid α
          a₁ b₁ : α
          a₂ b₂ : Subtype fun x => Membership.mem s x
          a✝ b✝ c✝ : Localization s
          a b c : Prod α (Subtype fun x => Membership.mem s x)
          hab : LE.le (HMul.hMul (↑b.2) a.1) (HMul.hMul (↑a.2) b.1)
          hbc : LE.le (HMul.hMul (↑c.2) b.1) (HMul.hMul (↑b.2) c.1)
          ⊢ α
        -/
      · exact ↑b.2
        /-
          🎉 no goals
        -/
      /-
        α : Type u_1
        inst✝ : OrderedCancelCommMonoid α
        s : Submonoid α
        a₁ b₁ : α
        a₂ b₂ : Subtype fun x => Membership.mem s x
        a✝ b✝ c✝ : Localization s
        a b c : Prod α (Subtype fun x => Membership.mem s x)
        hab : LE.le (HMul.hMul (↑b.2) a.1) (HMul.hMul (↑a.2) b.1)
        hbc : LE.le (HMul.hMul (↑c.2) b.1) (HMul.hMul (↑b.2) c.1)
        ⊢ LE.le (HMul.hMul (↑b.2) (HMul.hMul (↑c.2) a.1)) (HMul.hMul (↑b.2) (HMul.hMul …
      -/
      rw [mul_left_comm]
      /-
        α : Type u_1
        inst✝ : OrderedCancelCommMonoid α
        s : Submonoid α
        a₁ b₁ : α
        a₂ b₂ : Subtype fun x => Membership.mem s x
        a✝ b✝ c✝ : Localization s
        a b c : Prod α (Subtype fun x => Membership.mem s x)
        hab : LE.le (HMul.hMul (↑b.2) a.1) (HMul.hMul (↑a.2) b.1)
        hbc : LE.le (HMul.hMul (↑c.2) b.1) (HMul.hMul (↑b.2) c.1)
        ⊢ LE.le (HMul.hMul (↑c.2) (HMul.hMul (↑b.2) a.1)) (HMul.hMul (↑b.2) (HMul.hMul …
      -/
      refine (mul_le_mul_left' hab _).trans ?_
      /-
        α : Type u_1
        inst✝ : OrderedCancelCommMonoid α
        s : Submonoid α
        a₁ b₁ : α
        a₂ b₂ : Subtype fun x => Membership.mem s x
        a✝ b✝ c✝ : Localization s
        a b c : Prod α (Subtype fun x => Membership.mem s x)
        hab : LE.le (HMul.hMul (↑b.2) a.1) (HMul.hMul (↑a.2) b.1)
        hbc : LE.le (HMul.hMul (↑c.2) b.1) (HMul.hMul (↑b.2) c.1)
        ⊢ LE.le (HMul.hMul (↑c.2) (HMul.hMul (↑a.2) b.1)) (HMul.hMul (↑b.2) (HMul.hMul …
      -/
      rwa [mul_left_comm, mul_left_comm (b.2 : α), mul_le_mul_iff_left]
      /-
        🎉 no goals
      -/
  le_antisymm a b := by
    /-
      α : Type u_1
      inst✝ : OrderedCancelCommMonoid α
      s : Submonoid α
      a₁ b₁ : α
      a₂ b₂ : Subtype fun x => Membership.mem s x
      a b : Localization s
      ⊢ LE.le a b → LE.le b a → Eq a b
    -/
    induction' a using Localization.rec with a₁ a₂
    on_goal 1 =>
      induction' b using Localization.rec with b₁ b₂
      · simp_rw [mk_le_mk, mk_eq_mk_iff, r_iff_exists]
        exact fun hab hba => ⟨1, by rw [hab.antisymm hba]⟩
    /-
      case f.H
      α : Type u_1
      inst✝ : OrderedCancelCommMonoid α
      s : Submonoid α
      a₁✝ b₁ : α
      a₂✝ b₂ : Subtype fun x => Membership.mem s x
      b : Localization s
      a₁ : α
      a₂ : Subtype fun x => Membership.mem s x
      a✝ c✝ : α
      b✝ d✝ : Subtype fun x => Membership.mem s x
      h✝ : (Localization.r s) { fst := a✝, snd := b✝ } { fst := c✝, snd := d✝ }
      ⊢ Eq ⋯ ⋯
    -/
    all_goals rfl
    /-
      🎉 no goals
    -/
  lt_iff_le_not_le a b := Localization.induction_on₂ a b fun _ _ => lt_iff_le_not_le


@[to_additive]
instance orderedCancelCommMonoid : OrderedCancelCommMonoid (Localization s) where
  mul_le_mul_left := fun a b =>
    Localization.induction_on₂ a b fun a b hab c =>
      Localization.induction_on c fun c => by
        /-
          α : Type u_1
          inst✝ : OrderedCancelCommMonoid α
          s : Submonoid α
          a₁ b₁ : α
          a₂ b₂ : Subtype fun x => Membership.mem s x
          a✝ b✝ : Localization s
          a b : Prod α (Subtype fun x => Membership.mem s x)
          hab : LE.le (Localization.mk a.1 a.2) (Localization.mk b.1 b.2)
          c✝ : Localization s
          c : Prod α (Subtype fun x => Membership.mem s x)
          ⊢ LE.le (HMul.hMul (Localization.mk c.1 c.2) (Localization.mk a.1 a.2)) (HMul. …
        -/
        simp only [mk_mul, mk_le_mk, Submonoid.coe_mul, mul_mul_mul_comm _ _ c.1] at hab ⊢
        /-
          α : Type u_1
          inst✝ : OrderedCancelCommMonoid α
          s : Submonoid α
          a₁ b₁ : α
          a₂ b₂ : Subtype fun x => Membership.mem s x
          a✝ b✝ : Localization s
          a b : Prod α (Subtype fun x => Membership.mem s x)
          c✝ : Localization s
          c : Prod α (Subtype fun x => Membership.mem s x)
          hab : LE.le (HMul.hMul (↑b.2) a.1) (HMul.hMul (↑a.2) b.1)
          ⊢ LE.le (HMul.hMul (HMul.hMul (↑c.2) c.1) (HMul.hMul (↑b.2) a.1)) (HMul.hMul ( …
        -/
        exact mul_le_mul_left' hab _
        /-
          🎉 no goals
        -/
  le_of_mul_le_mul_left := fun a b c =>
    Localization.induction_on₃ a b c fun a b c hab => by
      /-
        α : Type u_1
        inst✝ : OrderedCancelCommMonoid α
        s : Submonoid α
        a₁ b₁ : α
        a₂ b₂ : Subtype fun x => Membership.mem s x
        a✝ b✝ c✝ : Localization s
        a b c : Prod α (Subtype fun x => Membership.mem s x)
        hab : LE.le (HMul.hMul (Localization.mk a.1 a.2) (Localization.mk b.1 b.2)) (H …
        ⊢ LE.le (Localization.mk b.1 b.2) (Localization.mk c.1 c.2)
      -/
      simp only [mk_mul, mk_le_mk, Submonoid.coe_mul, mul_mul_mul_comm _ _ a.1] at hab ⊢
      /-
        α : Type u_1
        inst✝ : OrderedCancelCommMonoid α
        s : Submonoid α
        a₁ b₁ : α
        a₂ b₂ : Subtype fun x => Membership.mem s x
        a✝ b✝ c✝ : Localization s
        a b c : Prod α (Subtype fun x => Membership.mem s x)
        hab : LE.le (HMul.hMul (HMul.hMul (↑a.2) a.1) (HMul.hMul (↑c.2) b.1)) (HMul.hM …
        ⊢ LE.le (HMul.hMul (↑c.2) b.1) (HMul.hMul (↑b.2) c.1)
      -/
      exact le_of_mul_le_mul_left' hab
      /-
        🎉 no goals
      -/


@[to_additive]
instance decidableLE [DecidableRel ((· ≤ ·) : α → α → Prop)] :
    DecidableRel ((· ≤ ·) : Localization s → Localization s → Prop) := fun a b =>
  Localization.recOnSubsingleton₂ a b fun _ _ _ _ => decidable_of_iff' _ mk_le_mk


@[to_additive]
instance decidableLT [DecidableRel ((· < ·) : α → α → Prop)] :
    DecidableRel ((· < ·) : Localization s → Localization s → Prop) := fun a b =>
  Localization.recOnSubsingleton₂ a b fun _ _ _ _ => decidable_of_iff' _ mk_lt_mk


/-- An ordered cancellative monoid injects into its localization by sending `a` to `a / b`. -/
@[to_additive (attr := simps!) "An ordered cancellative monoid injects into its localization by
sending `a` to `a - b`."]
def mkOrderEmbedding (b : s) : α ↪o Localization s where
  toFun a := mk a b
  inj' := mk_left_injective _
                           /-
                             α : Type u_1
                             inst✝ : OrderedCancelCommMonoid α
                             s : Submonoid α
                             a₁ b₁ : α
                             a₂ b₂ b✝ : Subtype fun x => Membership.mem s x
                             a b : α
                             ⊢ Iff (LE.le ({ toFun := fun a => Localization.mk a b✝, inj' := ⋯ } a) ({ toFu …
                           -/
  map_rel_iff' {a b} := by simp [mk_le_mk]
                           /-
                             🎉 no goals
                           -/


@[to_additive]
instance [LinearOrderedCancelCommMonoid α] {s : Submonoid α} :
    LinearOrderedCancelCommMonoid (Localization s) :=
  { Localization.orderedCancelCommMonoid with
    le_total := fun a b =>
      Localization.induction_on₂ a b fun _ _ => by
        /-
          α : Type u_1
          inst✝ : LinearOrderedCancelCommMonoid α
          s : Submonoid α
          a b : Localization s
          x✝¹ x✝ : Prod α (Subtype fun x => Membership.mem s x)
          ⊢ Or (LE.le (Localization.mk x✝¹.1 x✝¹.2) (Localization.mk x✝.1 x✝.2)) (LE.le  …
        -/
        simp_rw [mk_le_mk]
        /-
          α : Type u_1
          inst✝ : LinearOrderedCancelCommMonoid α
          s : Submonoid α
          a b : Localization s
          x✝¹ x✝ : Prod α (Subtype fun x => Membership.mem s x)
          ⊢ Or (LE.le (HMul.hMul (↑x✝.2) x✝¹.1) (HMul.hMul (↑x✝¹.2) x✝.1)) (LE.le (HMul. …
        -/
        exact le_total _ _
        /-
          🎉 no goals
        -/
    decidableLE := Localization.decidableLE
    decidableLT := Localization.decidableLT  -- Porting note: was wrong in mathlib3
    decidableEq := Localization.decidableEq }


