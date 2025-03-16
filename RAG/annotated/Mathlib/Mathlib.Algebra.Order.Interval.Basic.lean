@[to_additive]
instance : One (NonemptyInterval α) :=
  ⟨NonemptyInterval.pure 1⟩


@[to_additive (attr := simp) toProd_zero]
theorem toProd_one : (1 : NonemptyInterval α).toProd = 1 :=
  rfl


@[to_additive]
theorem fst_one : (1 : NonemptyInterval α).fst = 1 :=
  rfl


@[to_additive]
theorem snd_one : (1 : NonemptyInterval α).snd = 1 :=
  rfl

-- Porting note: Originally `@[simp, norm_cast, to_additive]`

@[to_additive (attr := push_cast, simp)]
theorem coe_one_interval : ((1 : NonemptyInterval α) : Interval α) = 1 :=
  rfl


@[to_additive (attr := simp)]
theorem pure_one : pure (1 : α) = 1 :=
  rfl


@[to_additive] lemma one_ne_bot : (1 : Interval α) ≠ ⊥ := pure_ne_bot


@[to_additive] lemma bot_ne_one : (⊥ : Interval α) ≠ 1 := bot_ne_pure


@[to_additive (attr := simp)]
theorem coe_one : ((1 : NonemptyInterval α) : Set α) = 1 :=
  coe_pure _


@[to_additive]
theorem one_mem_one : (1 : α) ∈ (1 : NonemptyInterval α) :=
  ⟨le_rfl, le_rfl⟩


@[to_additive (attr := simp)]
theorem coe_one : ((1 : Interval α) : Set α) = 1 :=
  Icc_self _


@[to_additive]
theorem one_mem_one : (1 : α) ∈ (1 : Interval α) :=
  ⟨le_rfl, le_rfl⟩


@[to_additive]
instance : Mul (NonemptyInterval α) :=
  ⟨fun s t => ⟨s.toProd * t.toProd, mul_le_mul' s.fst_le_snd t.fst_le_snd⟩⟩


@[to_additive]
instance : Mul (Interval α) :=
  ⟨Option.map₂ (· * ·)⟩


@[to_additive (attr := simp) toProd_add]
theorem toProd_mul : (s * t).toProd = s.toProd * t.toProd :=
  rfl


@[to_additive]
theorem fst_mul : (s * t).fst = s.fst * t.fst :=
  rfl


@[to_additive]
theorem snd_mul : (s * t).snd = s.snd * t.snd :=
  rfl


@[to_additive (attr := simp)]
theorem coe_mul_interval : (↑(s * t) : Interval α) = s * t :=
  rfl


@[to_additive (attr := simp)]
theorem pure_mul_pure : pure a * pure b = pure (a * b) :=
  rfl


@[to_additive (attr := simp)]
theorem bot_mul : ⊥ * t = ⊥ :=
  rfl


@[to_additive]
theorem mul_bot : s * ⊥ = ⊥ :=
  Option.map₂_none_right _ _

-- Porting note: simp can prove `add_bot`

instance NonemptyInterval.hasNSMul [AddMonoid α] [Preorder α] [AddLeftMono α]
    [AddRightMono α] : SMul ℕ (NonemptyInterval α) :=
  ⟨fun n s => ⟨(n • s.fst, n • s.snd), nsmul_le_nsmul_right s.fst_le_snd _⟩⟩


@[to_additive existing]
instance NonemptyInterval.hasPow [MulLeftMono α] [MulRightMono α] :
    Pow (NonemptyInterval α) ℕ :=
  ⟨fun s n => ⟨s.toProd ^ n, pow_le_pow_left' s.fst_le_snd _⟩⟩


@[to_additive (attr := simp) toProd_nsmul]
theorem toProd_pow : (s ^ n).toProd = s.toProd ^ n :=
  rfl


@[to_additive]
theorem fst_pow : (s ^ n).fst = s.fst ^ n :=
  rfl


@[to_additive]
theorem snd_pow : (s ^ n).snd = s.snd ^ n :=
  rfl


@[to_additive (attr := simp)]
theorem pure_pow : pure a ^ n = pure (a ^ n) :=
  rfl


@[to_additive]
instance commMonoid [OrderedCommMonoid α] : CommMonoid (NonemptyInterval α) :=
  NonemptyInterval.toProd_injective.commMonoid _ toProd_one toProd_mul toProd_pow


@[to_additive]
instance Interval.mulOneClass [OrderedCommMonoid α] : MulOneClass (Interval α) where
  mul := (· * ·)
  one := 1
  one_mul s :=
    (Option.map₂_coe_left _ _ _).trans <| by
      /-
        ι : Type u_1
        α : Type u_2
        inst✝ : OrderedCommMonoid α
        s : Interval α
        ⊢ Eq (Option.map (fun b => HMul.hMul 1 b) s) s
      -/
      simp_rw [one_mul, ← Function.id_def, Option.map_id, id]
      /-
        🎉 no goals
      -/
  mul_one s :=
    (Option.map₂_coe_right _ _ _).trans <| by
      /-
        ι : Type u_1
        α : Type u_2
        inst✝ : OrderedCommMonoid α
        s : Interval α
        ⊢ Eq (Option.map (fun a => HMul.hMul a 1) s) s
      -/
      simp_rw [mul_one, ← Function.id_def, Option.map_id, id]
      /-
        🎉 no goals
      -/


@[to_additive]
instance Interval.commMonoid [OrderedCommMonoid α] : CommMonoid (Interval α) :=
  { Interval.mulOneClass with
    mul_comm := fun _ _ => Option.map₂_comm mul_comm
    mul_assoc := fun _ _ _ => Option.map₂_assoc mul_assoc }


@[to_additive]
theorem coe_pow_interval [OrderedCommMonoid α] (s : NonemptyInterval α) (n : ℕ) :
    ↑(s ^ n) = (s : Interval α) ^ n :=
  map_pow (⟨⟨(↑), coe_one_interval⟩, coe_mul_interval⟩ : NonemptyInterval α →* Interval α) _ _

-- Porting note: simp can prove `coe_nsmul_interval`

@[to_additive]
theorem bot_pow : ∀ {n : ℕ}, n ≠ 0 → (⊥ : Interval α) ^ n = ⊥
  | 0, h => (h rfl).elim
  | Nat.succ n, _ => mul_bot (⊥ ^ n)


instance : Sub (NonemptyInterval α) :=
  ⟨fun s t => ⟨(s.fst - t.snd, s.snd - t.fst), tsub_le_tsub s.fst_le_snd t.fst_le_snd⟩⟩


instance : Sub (Interval α) :=
  ⟨Option.map₂ Sub.sub⟩


@[simp]
theorem fst_sub : (s - t).fst = s.fst - t.snd :=
  rfl


@[simp]
theorem snd_sub : (s - t).snd = s.snd - t.fst :=
  rfl


@[simp]
theorem coe_sub_interval : (↑(s - t) : Interval α) = s - t :=
  rfl


theorem sub_mem_sub (ha : a ∈ s) (hb : b ∈ t) : a - b ∈ s - t :=
  ⟨tsub_le_tsub ha.1 hb.2, tsub_le_tsub ha.2 hb.1⟩


@[simp]
theorem pure_sub_pure (a b : α) : pure a - pure b = pure (a - b) :=
  rfl


@[simp]
theorem bot_sub : ⊥ - t = ⊥ :=
  rfl


@[simp]
theorem sub_bot : s - ⊥ = ⊥ :=
  Option.map₂_none_right _ _


@[to_additive existing]
instance : Div (NonemptyInterval α) :=
  ⟨fun s t => ⟨(s.fst / t.snd, s.snd / t.fst), div_le_div'' s.fst_le_snd t.fst_le_snd⟩⟩


@[to_additive existing]
instance : Div (Interval α) :=
  ⟨Option.map₂ (· / ·)⟩


@[to_additive existing (attr := simp)]
theorem fst_div : (s / t).fst = s.fst / t.snd :=
  rfl


@[to_additive existing (attr := simp)]
theorem snd_div : (s / t).snd = s.snd / t.fst :=
  rfl


@[to_additive existing (attr := simp)]
theorem coe_div_interval : (↑(s / t) : Interval α) = s / t :=
  rfl


@[to_additive existing]
theorem div_mem_div (ha : a ∈ s) (hb : b ∈ t) : a / b ∈ s / t :=
  ⟨div_le_div'' ha.1 hb.2, div_le_div'' ha.2 hb.1⟩


@[to_additive existing (attr := simp)]
theorem pure_div_pure : pure a / pure b = pure (a / b) :=
  rfl


@[to_additive existing (attr := simp)]
theorem bot_div : ⊥ / t = ⊥ :=
  rfl


@[to_additive existing (attr := simp)]
theorem div_bot : s / ⊥ = ⊥ :=
  Option.map₂_none_right _ _


@[to_additive]
instance : Inv (NonemptyInterval α) :=
  ⟨fun s => ⟨(s.snd⁻¹, s.fst⁻¹), inv_le_inv' s.fst_le_snd⟩⟩


@[to_additive]
instance : Inv (Interval α) :=
  ⟨Option.map Inv.inv⟩


@[to_additive (attr := simp)]
theorem fst_inv : s⁻¹.fst = s.snd⁻¹ :=
  rfl


@[to_additive (attr := simp)]
theorem snd_inv : s⁻¹.snd = s.fst⁻¹ :=
  rfl


@[to_additive (attr := simp)]
theorem coe_inv_interval : (↑(s⁻¹) : Interval α) = (↑s)⁻¹ :=
  rfl


@[to_additive]
theorem inv_mem_inv (ha : a ∈ s) : a⁻¹ ∈ s⁻¹ :=
  ⟨inv_le_inv' ha.2, inv_le_inv' ha.1⟩


@[to_additive (attr := simp)]
theorem inv_pure : (pure a)⁻¹ = pure a⁻¹ :=
  rfl


@[to_additive (attr := simp)]
theorem Interval.inv_bot : (⊥ : Interval α)⁻¹ = ⊥ :=
  rfl


@[to_additive]
protected theorem mul_eq_one_iff : s * t = 1 ↔ ∃ a b, s = pure a ∧ t = pure b ∧ a * b = 1 := by
  /-
    α : Type u_2
    inst✝ : OrderedCommGroup α
    s t : NonemptyInterval α
    ⊢ Iff (Eq (HMul.hMul s t) 1) (Exists fun a => Exists fun b => And (Eq s (Nonem …
  -/
  refine ⟨fun h => ?_, ?_⟩
    /-
      case refine_1
      α : Type u_2
      inst✝ : OrderedCommGroup α
      s t : NonemptyInterval α
      h : Eq (HMul.hMul s t) 1
      ⊢ Exists fun a => Exists fun b => And (Eq s (NonemptyInterval.pure a)) (And (E …
    -/
  · rw [NonemptyInterval.ext_iff, Prod.ext_iff] at h
    /-
      case refine_1
      α : Type u_2
      inst✝ : OrderedCommGroup α
      s t : NonemptyInterval α
      h : And (Eq (HMul.hMul s t).toProd.1 (NonemptyInterval.toProd 1).1) (Eq (HMul. …
      ⊢ Exists fun a => Exists fun b => And (Eq s (NonemptyInterval.pure a)) (And (E …
    -/
    have := (mul_le_mul_iff_of_ge s.fst_le_snd t.fst_le_snd).1 (h.2.trans h.1.symm).le
    /-
      case refine_1
      α : Type u_2
      inst✝ : OrderedCommGroup α
      s t : NonemptyInterval α
      h : And (Eq (HMul.hMul s t).toProd.1 (NonemptyInterval.toProd 1).1) (Eq (HMul. …
      this : And (Eq s.toProd.1 s.toProd.2) (Eq t.toProd.1 t.toProd.2)
      ⊢ Exists fun a => Exists fun b => And (Eq s (NonemptyInterval.pure a)) (And (E …
    -/
    refine ⟨s.fst, t.fst, ?_, ?_, h.1⟩ <;> apply NonemptyInterval.ext <;> dsimp [pure]
      /-
        case refine_1.refine_1.toProd
        α : Type u_2
        inst✝ : OrderedCommGroup α
        s t : NonemptyInterval α
        h : And (Eq (HMul.hMul s t).toProd.1 (NonemptyInterval.toProd 1).1) (Eq (HMul. …
        this : And (Eq s.toProd.1 s.toProd.2) (Eq t.toProd.1 t.toProd.2)
        ⊢ Eq s.toProd { fst := s.toProd.1, snd := s.toProd.1 }
      -/
    · nth_rw 2 [this.1]
      /-
        🎉 no goals
      -/
      /-
        case refine_1.refine_2.toProd
        α : Type u_2
        inst✝ : OrderedCommGroup α
        s t : NonemptyInterval α
        h : And (Eq (HMul.hMul s t).toProd.1 (NonemptyInterval.toProd 1).1) (Eq (HMul. …
        this : And (Eq s.toProd.1 s.toProd.2) (Eq t.toProd.1 t.toProd.2)
        ⊢ Eq t.toProd { fst := t.toProd.1, snd := t.toProd.1 }
      -/
    · nth_rw 2 [this.2]
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      α : Type u_2
      inst✝ : OrderedCommGroup α
      s t : NonemptyInterval α
      ⊢ (Exists fun a => Exists fun b => And (Eq s (NonemptyInterval.pure a)) (And ( …
    -/
  · rintro ⟨b, c, rfl, rfl, h⟩
    /-
      case refine_2.intro.intro.intro.intro
      α : Type u_2
      inst✝ : OrderedCommGroup α
      b c : α
      h : Eq (HMul.hMul b c) 1
      ⊢ Eq (HMul.hMul (NonemptyInterval.pure b) (NonemptyInterval.pure c)) 1
    -/
    rw [pure_mul_pure, h, pure_one]
    /-
      🎉 no goals
    -/


instance subtractionCommMonoid {α : Type u} [OrderedAddCommGroup α] :
    SubtractionCommMonoid (NonemptyInterval α) :=
  { NonemptyInterval.addCommMonoid with
    neg := Neg.neg
    sub := Sub.sub
    sub_eq_add_neg := fun s t => by
      /-
        ι : Type u_1
        α✝ : Type u_2
        inst✝¹ : OrderedCommGroup α✝
        s✝ t✝ : NonemptyInterval α✝
        α : Type u
        inst✝ : OrderedAddCommGroup α
        s t : NonemptyInterval α
        ⊢ Eq (HSub.hSub s t) (HAdd.hAdd s (Neg.neg t))
      -/
      refine NonemptyInterval.ext (Prod.ext ?_ ?_) <;>
      /-
        case refine_1
        ι : Type u_1
        α✝ : Type u_2
        inst✝¹ : OrderedCommGroup α✝
        s✝ t✝ : NonemptyInterval α✝
        α : Type u
        inst✝ : OrderedAddCommGroup α
        s t : NonemptyInterval α
        ⊢ Eq (HSub.hSub s t).toProd.1 (HAdd.hAdd s (Neg.neg t)).toProd.1
      -/
      /-
        🎉 no goals
      -/
      exact sub_eq_add_neg _ _
      /-
        🎉 no goals
      -/
                           /-
                             ι : Type u_1
                             α✝ : Type u_2
                             inst✝¹ : OrderedCommGroup α✝
                             s✝ t : NonemptyInterval α✝
                             α : Type u
                             inst✝ : OrderedAddCommGroup α
                             s : NonemptyInterval α
                             ⊢ Eq (Neg.neg (Neg.neg s)) s
                           -/
    neg_neg := fun s => by apply NonemptyInterval.ext; exact neg_neg _
                                                       /-
                                                         🎉 no goals
                                                       -/
    neg_add_rev := fun s t => by
      /-
        ι : Type u_1
        α✝ : Type u_2
        inst✝¹ : OrderedCommGroup α✝
        s✝ t✝ : NonemptyInterval α✝
        α : Type u
        inst✝ : OrderedAddCommGroup α
        s t : NonemptyInterval α
        ⊢ Eq (Neg.neg (HAdd.hAdd s t)) (HAdd.hAdd (Neg.neg t) (Neg.neg s))
      -/
      refine NonemptyInterval.ext (Prod.ext ?_ ?_) <;>
      /-
        case refine_1
        ι : Type u_1
        α✝ : Type u_2
        inst✝¹ : OrderedCommGroup α✝
        s✝ t✝ : NonemptyInterval α✝
        α : Type u
        inst✝ : OrderedAddCommGroup α
        s t : NonemptyInterval α
        ⊢ Eq (Neg.neg (HAdd.hAdd s t)).toProd.1 (HAdd.hAdd (Neg.neg t) (Neg.neg s)).to …
      -/
      /-
        🎉 no goals
      -/
      exact neg_add_rev _ _
      /-
        🎉 no goals
      -/
    neg_eq_of_add := fun s t h => by
      /-
        ι : Type u_1
        α✝ : Type u_2
        inst✝¹ : OrderedCommGroup α✝
        s✝ t✝ : NonemptyInterval α✝
        α : Type u
        inst✝ : OrderedAddCommGroup α
        s t : NonemptyInterval α
        h : Eq (HAdd.hAdd s t) 0
        ⊢ Eq (Neg.neg s) t
      -/
      obtain ⟨a, b, rfl, rfl, hab⟩ := NonemptyInterval.add_eq_zero_iff.1 h
      /-
        case intro.intro.intro.intro
        ι : Type u_1
        α✝ : Type u_2
        inst✝¹ : OrderedCommGroup α✝
        s t : NonemptyInterval α✝
        α : Type u
        inst✝ : OrderedAddCommGroup α
        a b : α
        hab : Eq (HAdd.hAdd a b) 0
        h : Eq (HAdd.hAdd (NonemptyInterval.pure a) (NonemptyInterval.pure b)) 0
        ⊢ Eq (Neg.neg (NonemptyInterval.pure a)) (NonemptyInterval.pure b)
      -/
      rw [neg_pure, neg_eq_of_add_eq_zero_right hab]
      /-
        🎉 no goals
      -/
    -- TODO: use a better defeq
    zsmul := zsmulRec }


@[to_additive existing NonemptyInterval.subtractionCommMonoid]
instance divisionCommMonoid : DivisionCommMonoid (NonemptyInterval α) :=
  { NonemptyInterval.commMonoid with
    inv := Inv.inv
    div := (· / ·)
    div_eq_mul_inv := fun s t => by
      /-
        ι : Type u_1
        α : Type u_2
        inst✝ : OrderedCommGroup α
        s✝ t✝ s t : NonemptyInterval α
        ⊢ Eq (HDiv.hDiv s t) (HMul.hMul s (Inv.inv t))
      -/
      refine NonemptyInterval.ext (Prod.ext ?_ ?_) <;>
      /-
        case refine_1
        ι : Type u_1
        α : Type u_2
        inst✝ : OrderedCommGroup α
        s✝ t✝ s t : NonemptyInterval α
        ⊢ Eq (HDiv.hDiv s t).toProd.1 (HMul.hMul s (Inv.inv t)).toProd.1
      -/
      /-
        🎉 no goals
      -/
      exact div_eq_mul_inv _ _
      /-
        🎉 no goals
      -/
                           /-
                             ι : Type u_1
                             α : Type u_2
                             inst✝ : OrderedCommGroup α
                             s✝ t s : NonemptyInterval α
                             ⊢ Eq (Inv.inv (Inv.inv s)) s
                           -/
    inv_inv := fun s => by apply NonemptyInterval.ext; exact inv_inv _
                                                       /-
                                                         🎉 no goals
                                                       -/
    mul_inv_rev := fun s t => by
      /-
        ι : Type u_1
        α : Type u_2
        inst✝ : OrderedCommGroup α
        s✝ t✝ s t : NonemptyInterval α
        ⊢ Eq (Inv.inv (HMul.hMul s t)) (HMul.hMul (Inv.inv t) (Inv.inv s))
      -/
      refine NonemptyInterval.ext (Prod.ext ?_ ?_) <;>
      /-
        case refine_1
        ι : Type u_1
        α : Type u_2
        inst✝ : OrderedCommGroup α
        s✝ t✝ s t : NonemptyInterval α
        ⊢ Eq (Inv.inv (HMul.hMul s t)).toProd.1 (HMul.hMul (Inv.inv t) (Inv.inv s)).to …
      -/
      /-
        🎉 no goals
      -/
      exact mul_inv_rev _ _
      /-
        🎉 no goals
      -/
    inv_eq_of_mul := fun s t h => by
      /-
        ι : Type u_1
        α : Type u_2
        inst✝ : OrderedCommGroup α
        s✝ t✝ s t : NonemptyInterval α
        h : Eq (HMul.hMul s t) 1
        ⊢ Eq (Inv.inv s) t
      -/
      obtain ⟨a, b, rfl, rfl, hab⟩ := NonemptyInterval.mul_eq_one_iff.1 h
      /-
        case intro.intro.intro.intro
        ι : Type u_1
        α : Type u_2
        inst✝ : OrderedCommGroup α
        s t : NonemptyInterval α
        a b : α
        hab : Eq (HMul.hMul a b) 1
        h : Eq (HMul.hMul (NonemptyInterval.pure a) (NonemptyInterval.pure b)) 1
        ⊢ Eq (Inv.inv (NonemptyInterval.pure a)) (NonemptyInterval.pure b)
      -/
      rw [inv_pure, inv_eq_of_mul_eq_one_right hab] }
      /-
        🎉 no goals
      -/


@[to_additive]
protected theorem mul_eq_one_iff : s * t = 1 ↔ ∃ a b, s = pure a ∧ t = pure b ∧ a * b = 1 := by
  /-
    α : Type u_2
    inst✝ : OrderedCommGroup α
    s t : Interval α
    ⊢ Iff (Eq (HMul.hMul s t) 1) (Exists fun a => Exists fun b => And (Eq s (Inter …
  -/
  cases s
    /-
      case bot
      α : Type u_2
      inst✝ : OrderedCommGroup α
      t : Interval α
      ⊢ Iff (Eq (HMul.hMul Bot.bot t) 1) (Exists fun a => Exists fun b => And (Eq Bo …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case coe
    α : Type u_2
    inst✝ : OrderedCommGroup α
    t : Interval α
    a✝ : NonemptyInterval α
    ⊢ Iff (Eq (HMul.hMul (↑a✝) t) 1) (Exists fun a => Exists fun b => And (Eq (↑a✝ …
  -/
  cases t
    /-
      case coe.bot
      α : Type u_2
      inst✝ : OrderedCommGroup α
      a✝ : NonemptyInterval α
      ⊢ Iff (Eq (HMul.hMul (↑a✝) Bot.bot) 1) (Exists fun a => Exists fun b => And (E …
    -/
  · simp
    /-
      🎉 no goals
    -/
  · simp_rw [← NonemptyInterval.coe_mul_interval, ← NonemptyInterval.coe_one_interval,
      WithBot.coe_inj, NonemptyInterval.coe_eq_pure]
    /-
      case coe.coe
      α : Type u_2
      inst✝ : OrderedCommGroup α
      a✝¹ a✝ : NonemptyInterval α
      ⊢ Iff (Eq (HMul.hMul a✝¹ a✝) 1) (Exists fun a => Exists fun b => And (Eq a✝¹ ( …
    -/
    exact NonemptyInterval.mul_eq_one_iff
    /-
      🎉 no goals
    -/


instance subtractionCommMonoid {α : Type u} [OrderedAddCommGroup α] :
    SubtractionCommMonoid (Interval α) :=
  { Interval.addCommMonoid with
    neg := Neg.neg
    sub := Sub.sub
    sub_eq_add_neg := by
      /-
        ι : Type u_1
        α✝ : Type u_2
        inst✝¹ : OrderedCommGroup α✝
        s t : Interval α✝
        α : Type u
        inst✝ : OrderedAddCommGroup α
        ⊢ ∀ (a b : Interval α), Eq (HSub.hSub a b) (HAdd.hAdd a (Neg.neg b))
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
      rintro (_ | s) (_ | t) <;> first |rfl|exact congr_arg some (sub_eq_add_neg _ _)
                                 /-
                                   🎉 no goals
                                 -/
                  /-
                    ι : Type u_1
                    α✝ : Type u_2
                    inst✝¹ : OrderedCommGroup α✝
                    s t : Interval α✝
                    α : Type u
                    inst✝ : OrderedAddCommGroup α
                    ⊢ ∀ (x : Interval α), Eq (Neg.neg (Neg.neg x)) x
                  -/
                                     /-
                                       🎉 no goals
                                     -/
    neg_neg := by rintro (_ | s) <;> first |rfl|exact congr_arg some (neg_neg _)
                                     /-
                                       🎉 no goals
                                     -/
                      /-
                        ι : Type u_1
                        α✝ : Type u_2
                        inst✝¹ : OrderedCommGroup α✝
                        s t : Interval α✝
                        α : Type u
                        inst✝ : OrderedAddCommGroup α
                        ⊢ ∀ (a b : Interval α), Eq (Neg.neg (HAdd.hAdd a b)) (HAdd.hAdd (Neg.neg b) (N …
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
    neg_add_rev := by rintro (_ | s) (_ | t) <;> first |rfl|exact congr_arg some (neg_add_rev _ _)
                                                 /-
                                                   🎉 no goals
                                                 -/
    neg_eq_of_add := by
      /-
        ι : Type u_1
        α✝ : Type u_2
        inst✝¹ : OrderedCommGroup α✝
        s t : Interval α✝
        α : Type u
        inst✝ : OrderedAddCommGroup α
        ⊢ ∀ (a b : Interval α), Eq (HAdd.hAdd a b) 0 → Eq (Neg.neg a) b
      -/
      rintro (_ | s) (_ | t) h <;>
        first
          | cases h
          | exact congr_arg some (neg_eq_of_add_eq_zero_right <| Option.some_injective _ h)
    -- TODO: use a better defeq
    zsmul := zsmulRec }


@[to_additive existing Interval.subtractionCommMonoid]
instance divisionCommMonoid : DivisionCommMonoid (Interval α) :=
  { Interval.commMonoid with
    inv := Inv.inv
    div := (· / ·)
    div_eq_mul_inv := by
      /-
        ι : Type u_1
        α : Type u_2
        inst✝ : OrderedCommGroup α
        s t : Interval α
        ⊢ ∀ (a b : Interval α), Eq (HDiv.hDiv a b) (HMul.hMul a (Inv.inv b))
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
      rintro (_ | s) (_ | t) <;> first |rfl|exact congr_arg some (div_eq_mul_inv _ _)
                                 /-
                                   🎉 no goals
                                 -/
                  /-
                    ι : Type u_1
                    α : Type u_2
                    inst✝ : OrderedCommGroup α
                    s t : Interval α
                    ⊢ ∀ (x : Interval α), Eq (Inv.inv (Inv.inv x)) x
                  -/
                                     /-
                                       🎉 no goals
                                     -/
    inv_inv := by rintro (_ | s) <;> first |rfl|exact congr_arg some (inv_inv _)
                                     /-
                                       🎉 no goals
                                     -/
                      /-
                        ι : Type u_1
                        α : Type u_2
                        inst✝ : OrderedCommGroup α
                        s t : Interval α
                        ⊢ ∀ (a b : Interval α), Eq (Inv.inv (HMul.hMul a b)) (HMul.hMul (Inv.inv b) (I …
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
    mul_inv_rev := by rintro (_ | s) (_ | t) <;> first |rfl|exact congr_arg some (mul_inv_rev _ _)
                                                 /-
                                                   🎉 no goals
                                                 -/
    inv_eq_of_mul := by
      /-
        ι : Type u_1
        α : Type u_2
        inst✝ : OrderedCommGroup α
        s t : Interval α
        ⊢ ∀ (a b : Interval α), Eq (HMul.hMul a b) 1 → Eq (Inv.inv a) b
      -/
      rintro (_ | s) (_ | t) h <;>
        first
          | cases h
          | exact congr_arg some (inv_eq_of_mul_eq_one_right <| Option.some_injective _ h) }


/-- The length of an interval is its first component minus its second component. This measures the
accuracy of the approximation by an interval. -/
def length : α :=
  s.snd - s.fst


@[simp]
theorem length_nonneg : 0 ≤ s.length :=
  sub_nonneg_of_le s.fst_le_snd


@[simp]
theorem length_pure : (pure a).length = 0 :=
  sub_self _


@[simp]
theorem length_zero : (0 : NonemptyInterval α).length = 0 :=
  length_pure _


@[simp]
theorem length_neg : (-s).length = s.length :=
  neg_sub_neg _ _


@[simp]
theorem length_add : (s + t).length = s.length + t.length :=
  add_sub_add_comm _ _ _ _


@[simp]
                                                                /-
                                                                  α : Type u_2
                                                                  inst✝ : OrderedAddCommGroup α
                                                                  s t : NonemptyInterval α
                                                                  ⊢ Eq (HSub.hSub s t).length (HAdd.hAdd s.length t.length)
                                                                -/
theorem length_sub : (s - t).length = s.length + t.length := by simp [sub_eq_add_neg]
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[simp]
theorem length_sum (f : ι → NonemptyInterval α) (s : Finset ι) :
    (∑ i ∈ s, f i).length = ∑ i ∈ s, (f i).length :=
  map_sum (⟨⟨length, length_zero⟩, length_add⟩ : NonemptyInterval α →+ α) _ _


/-- The length of an interval is its first component minus its second component. This measures the
accuracy of the approximation by an interval. -/
def length : Interval α → α
  | ⊥ => 0
  | (s : NonemptyInterval α) => s.length


@[simp]
theorem length_nonneg : ∀ s : Interval α, 0 ≤ s.length
  | ⊥ => le_rfl
  | (s : NonemptyInterval α) => s.length_nonneg


@[simp]
theorem length_pure : (pure a).length = 0 :=
  NonemptyInterval.length_pure _


@[simp]
theorem length_zero : (0 : Interval α).length = 0 :=
  length_pure _


@[simp]
theorem length_neg : ∀ s : Interval α, (-s).length = s.length
  | ⊥ => rfl
  | (s : NonemptyInterval α) => s.length_neg


theorem length_add_le : ∀ s t : Interval α, (s + t).length ≤ s.length + t.length
               /-
                 α : Type u_2
                 inst✝ : OrderedAddCommGroup α
                 x✝ : Interval α
                 ⊢ LE.le (HAdd.hAdd Bot.bot x✝).length (HAdd.hAdd Bot.bot.length x✝.length)
               -/
  | ⊥, _ => by simp
               /-
                 🎉 no goals
               -/
               /-
                 α : Type u_2
                 inst✝ : OrderedAddCommGroup α
                 x✝ : Interval α
                 ⊢ LE.le (HAdd.hAdd x✝ Bot.bot).length (HAdd.hAdd x✝.length Bot.bot.length)
               -/
  | _, ⊥ => by simp
               /-
                 🎉 no goals
               -/
  | (s : NonemptyInterval α), (t : NonemptyInterval α) => (s.length_add t).le


theorem length_sub_le : (s - t).length ≤ s.length + t.length := by
  /-
    α : Type u_2
    inst✝ : OrderedAddCommGroup α
    s t : Interval α
    ⊢ LE.le (HSub.hSub s t).length (HAdd.hAdd s.length t.length)
  -/
  simpa [sub_eq_add_neg] using length_add_le s (-t)
  /-
    🎉 no goals
  -/


theorem length_sum_le (f : ι → Interval α) (s : Finset ι) :
    (∑ i ∈ s, f i).length ≤ ∑ i ∈ s, (f i).length :=
  Finset.le_sum_of_subadditive _ length_zero length_add_le _ _


/-- Extension for the `positivity` tactic: The length of an interval is always nonnegative. -/
@[positivity NonemptyInterval.length _]
def evalNonemptyIntervalLength : PositivityExt where
  eval {u _α} _ _ e := do
    let ~q(@NonemptyInterval.length _ $inst $a) := e | throwError "not NonemptyInterval.length"
    assertInstancesCommute
    return .nonnegative q(NonemptyInterval.length_nonneg $a)


/-- Extension for the `positivity` tactic: The length of an interval is always nonnegative. -/
@[positivity Interval.length _]
def evalIntervalLength : PositivityExt where
  eval {u _α} _ _ e := do
    let ~q(@Interval.length _ $inst $a) := e | throwError "not Interval.length"
    assumeInstancesCommute
    return .nonnegative q(Interval.length_nonneg $a)


