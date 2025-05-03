/-- When writing a lemma about `someSetoid x y` (which uses this instance),
call it `someSetoid_apply` not `someSetoid_r`. -/
instance : CoeFun (Setoid α) (fun _ ↦ α → α → Prop) where
  coe := @Setoid.r _


theorem ext {α : Sort*} : ∀ {s t : Setoid α}, (∀ a b, s a b ↔ t a b) → s = t
  | ⟨r, _⟩, ⟨p, _⟩, Eq =>
     /-
       α : Sort u_3
       r : α → α → Prop
       iseqv✝¹ : Equivalence r
       p : α → α → Prop
       iseqv✝ : Equivalence p
       Eq : ∀ (a b : α), Iff ({ r := r, iseqv := iseqv✝¹ } a b) ({ r := p, iseqv := i …
       ⊢ _root_.Eq { r := r, iseqv := iseqv✝¹ } { r := p, iseqv := iseqv✝ }
     -/
  by have : r = p := funext fun a ↦ funext fun b ↦ propext <| Eq a b
     /-
       α : Sort u_3
       r : α → α → Prop
       iseqv✝¹ : Equivalence r
       p : α → α → Prop
       iseqv✝ : Equivalence p
       Eq : ∀ (a b : α), Iff ({ r := r, iseqv := iseqv✝¹ } a b) ({ r := p, iseqv := i …
       this : _root_.Eq r p
       ⊢ _root_.Eq { r := r, iseqv := iseqv✝¹ } { r := p, iseqv := iseqv✝ }
     -/
     subst this
     /-
       α : Sort u_3
       r : α → α → Prop
       iseqv✝¹ iseqv✝ : Equivalence r
       Eq : ∀ (a b : α), Iff ({ r := r, iseqv := iseqv✝¹ } a b) ({ r := r, iseqv := i …
       ⊢ _root_.Eq { r := r, iseqv := iseqv✝¹ } { r := r, iseqv := iseqv✝ }
     -/
     rfl
     /-
       🎉 no goals
     -/


@[inherit_doc Quot.mk]
local notation3:arg "⟦" a "⟧" => Quot.mk _ a


@[elab_as_elim]
protected theorem induction_on {α : Sort*} {r : α → α → Prop} {β : Quot r → Prop} (q : Quot r)
    (h : ∀ a, β (Quot.mk r a)) : β q :=
  ind h q


instance (r : α → α → Prop) [Inhabited α] : Inhabited (Quot r) :=
  ⟨⟦default⟧⟩


protected instance Subsingleton [Subsingleton α] : Subsingleton (Quot ra) :=
  ⟨fun x ↦ Quot.induction_on x fun _ ↦ Quot.ind fun _ ↦ congr_arg _ (Subsingleton.elim _ _)⟩


@[deprecated (since := "2024-08-26")] alias recOn' := Quot.recOn

@[deprecated (since := "2024-08-26")] alias recOnSubsingleton' := Quot.recOnSubsingleton


instance [Unique α] : Unique (Quot ra) := Unique.mk' _


/-- Recursion on two `Quotient` arguments `a` and `b`, result type depends on `⟦a⟧` and `⟦b⟧`. -/
protected def hrecOn₂ (qa : Quot ra) (qb : Quot rb) (f : ∀ a b, φ ⟦a⟧ ⟦b⟧)
    (ca : ∀ {b a₁ a₂}, ra a₁ a₂ → HEq (f a₁ b) (f a₂ b))
    (cb : ∀ {a b₁ b₂}, rb b₁ b₂ → HEq (f a b₁) (f a b₂)) :
    φ qa qb :=
  Quot.hrecOn (motive := fun qa ↦ φ qa qb) qa
    (fun a ↦ Quot.hrecOn qb (f a) (fun _ _ pb ↦ cb pb))
    fun a₁ a₂ pa ↦
      Quot.induction_on qb fun b ↦
        have h₁ : HEq (@Quot.hrecOn _ _ (φ _) ⟦b⟧ (f a₁) (@cb _)) (f a₁ b) := by
          /-
            α : Sort u_1
            β : Sort u_2
            ra : α → α → Prop
            rb : β → β → Prop
            φ : Quot ra → Quot rb → Sort u_3
            qa : Quot ra
            qb : Quot rb
            f : (a : α) → (b : β) → φ (Quot.mk ra a) (Quot.mk rb b)
            ca : ∀ {b : β} {a₁ a₂ : α}, ra a₁ a₂ → HEq (f a₁ b) (f a₂ b)
            cb : ∀ {a : α} {b₁ b₂ : β}, rb b₁ b₂ → HEq (f a b₁) (f a b₂)
            a₁ a₂ : α
            pa : ra a₁ a₂
            b : β
            ⊢ HEq (Quot.hrecOn (Quot.mk rb b) (f a₁) ⋯) (f a₁ b)
          -/
          simp [heq_self_iff_true]
          /-
            🎉 no goals
          -/
        have h₂ : HEq (f a₂ b) (@Quot.hrecOn _ _ (φ _) ⟦b⟧ (f a₂) (@cb _)) := by
          /-
            α : Sort u_1
            β : Sort u_2
            ra : α → α → Prop
            rb : β → β → Prop
            φ : Quot ra → Quot rb → Sort u_3
            qa : Quot ra
            qb : Quot rb
            f : (a : α) → (b : β) → φ (Quot.mk ra a) (Quot.mk rb b)
            ca : ∀ {b : β} {a₁ a₂ : α}, ra a₁ a₂ → HEq (f a₁ b) (f a₂ b)
            cb : ∀ {a : α} {b₁ b₂ : β}, rb b₁ b₂ → HEq (f a b₁) (f a b₂)
            a₁ a₂ : α
            pa : ra a₁ a₂
            b : β
            h₁ : HEq (Quot.hrecOn (Quot.mk rb b) (f a₁) ⋯) (f a₁ b)
            ⊢ HEq (f a₂ b) (Quot.hrecOn (Quot.mk rb b) (f a₂) ⋯)
          -/
          simp [heq_self_iff_true]
          /-
            🎉 no goals
          -/
        (h₁.trans (ca pa)).trans h₂


/-- Map a function `f : α → β` such that `ra x y` implies `rb (f x) (f y)`
to a map `Quot ra → Quot rb`. -/
protected def map (f : α → β) (h : ∀ ⦃a b : α⦄, ra a b → rb (f a) (f b)) : Quot ra → Quot rb :=
  Quot.lift (fun x => Quot.mk rb (f x)) fun _ _ hra ↦ Quot.sound <| h hra


/-- If `ra` is a subrelation of `ra'`, then we have a natural map `Quot ra → Quot ra'`. -/
protected def mapRight {ra' : α → α → Prop} (h : ∀ a₁ a₂, ra a₁ a₂ → ra' a₁ a₂) :
    Quot ra → Quot ra' :=
  Quot.map id h


/-- Weaken the relation of a quotient. This is the same as `Quot.map id`. -/
def factor {α : Type*} (r s : α → α → Prop) (h : ∀ x y, r x y → s x y) : Quot r → Quot s :=
  Quot.lift (Quot.mk s) fun x y rxy ↦ Quot.sound (h x y rxy)


theorem factor_mk_eq {α : Type*} (r s : α → α → Prop) (h : ∀ x y, r x y → s x y) :
    factor r s h ∘ Quot.mk _ = Quot.mk _ :=
  rfl


theorem lift_mk (f : α → γ) (h : ∀ a₁ a₂, r a₁ a₂ → f a₁ = f a₂) (a : α) :
    Quot.lift f h (Quot.mk r a) = f a :=
  rfl


theorem liftOn_mk (a : α) (f : α → γ) (h : ∀ a₁ a₂, r a₁ a₂ → f a₁ = f a₂) :
    Quot.liftOn (Quot.mk r a) f h = f a :=
  rfl


@[simp] theorem surjective_lift {f : α → γ} (h : ∀ a₁ a₂, r a₁ a₂ → f a₁ = f a₂) :
    Function.Surjective (lift f h) ↔ Function.Surjective f :=
  ⟨fun hf => hf.comp Quot.exists_rep, fun hf y => let ⟨x, hx⟩ := hf y; ⟨Quot.mk _ x, hx⟩⟩


/-- Descends a function `f : α → β → γ` to quotients of `α` and `β`. -/
protected def lift₂ (f : α → β → γ) (hr : ∀ a b₁ b₂, s b₁ b₂ → f a b₁ = f a b₂)
    (hs : ∀ a₁ a₂ b, r a₁ a₂ → f a₁ b = f a₂ b) (q₁ : Quot r) (q₂ : Quot s) : γ :=
  Quot.lift (fun a ↦ Quot.lift (f a) (hr a))
    (fun a₁ a₂ ha ↦ funext fun q ↦ Quot.induction_on q fun b ↦ hs a₁ a₂ b ha) q₁ q₂


@[simp]
theorem lift₂_mk (f : α → β → γ) (hr : ∀ a b₁ b₂, s b₁ b₂ → f a b₁ = f a b₂)
    (hs : ∀ a₁ a₂ b, r a₁ a₂ → f a₁ b = f a₂ b)
    (a : α) (b : β) : Quot.lift₂ f hr hs (Quot.mk r a) (Quot.mk s b) = f a b :=
  rfl


/-- Descends a function `f : α → β → γ` to quotients of `α` and `β` and applies it. -/
protected def liftOn₂ (p : Quot r) (q : Quot s) (f : α → β → γ)
    (hr : ∀ a b₁ b₂, s b₁ b₂ → f a b₁ = f a b₂) (hs : ∀ a₁ a₂ b, r a₁ a₂ → f a₁ b = f a₂ b) : γ :=
  Quot.lift₂ f hr hs p q


@[simp]
theorem liftOn₂_mk (a : α) (b : β) (f : α → β → γ) (hr : ∀ a b₁ b₂, s b₁ b₂ → f a b₁ = f a b₂)
    (hs : ∀ a₁ a₂ b, r a₁ a₂ → f a₁ b = f a₂ b) :
    Quot.liftOn₂ (Quot.mk r a) (Quot.mk s b) f hr hs = f a b :=
  rfl


/-- Descends a function `f : α → β → γ` to quotients of `α` and `β` with values in a quotient of
`γ`. -/
protected def map₂ (f : α → β → γ) (hr : ∀ a b₁ b₂, s b₁ b₂ → t (f a b₁) (f a b₂))
    (hs : ∀ a₁ a₂ b, r a₁ a₂ → t (f a₁ b) (f a₂ b)) (q₁ : Quot r) (q₂ : Quot s) : Quot t :=
  Quot.lift₂ (fun a b ↦ Quot.mk t <| f a b) (fun a b₁ b₂ hb ↦ Quot.sound (hr a b₁ b₂ hb))
    (fun a₁ a₂ b ha ↦ Quot.sound (hs a₁ a₂ b ha)) q₁ q₂


@[simp]
theorem map₂_mk (f : α → β → γ) (hr : ∀ a b₁ b₂, s b₁ b₂ → t (f a b₁) (f a b₂))
    (hs : ∀ a₁ a₂ b, r a₁ a₂ → t (f a₁ b) (f a₂ b)) (a : α) (b : β) :
    Quot.map₂ f hr hs (Quot.mk r a) (Quot.mk s b) = Quot.mk t (f a b) :=
  rfl


/-- A binary version of `Quot.recOnSubsingleton`. -/
@[elab_as_elim]
protected def recOnSubsingleton₂ {φ : Quot r → Quot s → Sort*}
    [h : ∀ a b, Subsingleton (φ ⟦a⟧ ⟦b⟧)] (q₁ : Quot r)
    (q₂ : Quot s) (f : ∀ a b, φ ⟦a⟧ ⟦b⟧) : φ q₁ q₂ :=
  @Quot.recOnSubsingleton _ r (fun q ↦ φ q q₂)
    (fun a ↦ Quot.ind (β := fun b ↦ Subsingleton (φ (mk r a) b)) (h a) q₂) q₁
    fun a ↦ Quot.recOnSubsingleton q₂ fun b ↦ f a b


@[elab_as_elim]
protected theorem induction_on₂ {δ : Quot r → Quot s → Prop} (q₁ : Quot r) (q₂ : Quot s)
    (h : ∀ a b, δ (Quot.mk r a) (Quot.mk s b)) : δ q₁ q₂ :=
  Quot.ind (β := fun a ↦ δ a q₂) (fun a₁ ↦ Quot.ind (fun a₂ ↦ h a₁ a₂) q₂) q₁


@[elab_as_elim]
protected theorem induction_on₃ {δ : Quot r → Quot s → Quot t → Prop} (q₁ : Quot r)
    (q₂ : Quot s) (q₃ : Quot t) (h : ∀ a b c, δ (Quot.mk r a) (Quot.mk s b) (Quot.mk t c)) :
    δ q₁ q₂ q₃ :=
  Quot.ind (β := fun a ↦ δ a q₂ q₃) (fun a₁ ↦ Quot.ind (β := fun b ↦ δ _ b q₃)
    (fun a₂ ↦ Quot.ind (fun a₃ ↦ h a₁ a₂ a₃) q₃) q₂) q₁


instance lift.decidablePred (r : α → α → Prop) (f : α → Prop) (h : ∀ a b, r a b → f a = f b)
    [hf : DecidablePred f] :
    DecidablePred (Quot.lift f h) :=
  fun q ↦ Quot.recOnSubsingleton (motive := fun _ ↦ Decidable _) q hf


/-- Note that this provides `DecidableRel (Quot.Lift₂ f ha hb)` when `α = β`. -/
instance lift₂.decidablePred (r : α → α → Prop) (s : β → β → Prop) (f : α → β → Prop)
    (ha : ∀ a b₁ b₂, s b₁ b₂ → f a b₁ = f a b₂) (hb : ∀ a₁ a₂ b, r a₁ a₂ → f a₁ b = f a₂ b)
    [hf : ∀ a, DecidablePred (f a)] (q₁ : Quot r) :
    DecidablePred (Quot.lift₂ f ha hb q₁) :=
  fun q₂ ↦ Quot.recOnSubsingleton₂ q₁ q₂ hf


instance (r : α → α → Prop) (q : Quot r) (f : α → Prop) (h : ∀ a b, r a b → f a = f b)
    [DecidablePred f] :
    Decidable (Quot.liftOn q f h) :=
  Quot.lift.decidablePred _ _ _ _


instance (r : α → α → Prop) (s : β → β → Prop) (q₁ : Quot r) (q₂ : Quot s) (f : α → β → Prop)
    (ha : ∀ a b₁ b₂, s b₁ b₂ → f a b₁ = f a b₂) (hb : ∀ a₁ a₂ b, r a₁ a₂ → f a₁ b = f a₂ b)
    [∀ a, DecidablePred (f a)] :
    Decidable (Quot.liftOn₂ q₁ q₂ f ha hb) :=
  Quot.lift₂.decidablePred _ _ _ _ _ _ _


@[inherit_doc Quotient.mk]
notation3:arg "⟦" a "⟧" => Quotient.mk _ a


instance instInhabitedQuotient (s : Setoid α) [Inhabited α] : Inhabited (Quotient s) :=
  ⟨⟦default⟧⟩


instance instSubsingletonQuotient (s : Setoid α) [Subsingleton α] : Subsingleton (Quotient s) :=
  Quot.Subsingleton


instance instUniqueQuotient (s : Setoid α) [Unique α] : Unique (Quotient s) := Unique.mk' _


instance {α : Type*} [Setoid α] : IsEquiv α (· ≈ ·) where
  refl := Setoid.refl
  symm _ _ := Setoid.symm
  trans _ _ _ := Setoid.trans


/-- Induction on two `Quotient` arguments `a` and `b`, result type depends on `⟦a⟧` and `⟦b⟧`. -/
protected def hrecOn₂ (qa : Quotient sa) (qb : Quotient sb) (f : ∀ a b, φ ⟦a⟧ ⟦b⟧)
    (c : ∀ a₁ b₁ a₂ b₂, a₁ ≈ a₂ → b₁ ≈ b₂ → HEq (f a₁ b₁) (f a₂ b₂)) : φ qa qb :=
  Quot.hrecOn₂ qa qb f (fun p ↦ c _ _ _ _ p (Setoid.refl _)) fun p ↦ c _ _ _ _ (Setoid.refl _) p


/-- Map a function `f : α → β` that sends equivalent elements to equivalent elements
to a function `Quotient sa → Quotient sb`. Useful to define unary operations on quotients. -/
protected def map (f : α → β) (h : ∀ ⦃a b : α⦄, a ≈ b → f a ≈ f b) : Quotient sa → Quotient sb :=
  Quot.map f h


@[simp]
theorem map_mk (f : α → β) (h) (x : α) :
    Quotient.map f h (⟦x⟧ : Quotient sa) = (⟦f x⟧ : Quotient sb) :=
  rfl


/-- Map a function `f : α → β → γ` that sends equivalent elements to equivalent elements
to a function `f : Quotient sa → Quotient sb → Quotient sc`.
Useful to define binary operations on quotients. -/
protected def map₂ (f : α → β → γ)
   (h : ∀ ⦃a₁ a₂⦄, a₁ ≈ a₂ → ∀ ⦃b₁ b₂⦄, b₁ ≈ b₂ → f a₁ b₁ ≈ f a₂ b₂) :
    Quotient sa → Quotient sb → Quotient sc :=
  Quotient.lift₂ (fun x y ↦ ⟦f x y⟧) fun _ _ _ _ h₁ h₂ ↦ Quot.sound <| h h₁ h₂


@[simp]
theorem map₂_mk (f : α → β → γ) (h) (x : α) (y : β) :
    Quotient.map₂ f h (⟦x⟧ : Quotient sa) (⟦y⟧ : Quotient sb) = (⟦f x y⟧ : Quotient sc) :=
  rfl


instance lift.decidablePred (f : α → Prop) (h : ∀ a b, a ≈ b → f a = f b) [DecidablePred f] :
    DecidablePred (Quotient.lift f h) :=
  Quot.lift.decidablePred _ _ _


/-- Note that this provides `DecidableRel (Quotient.lift₂ f h)` when `α = β`. -/
instance lift₂.decidablePred (f : α → β → Prop)
    (h : ∀ a₁ b₁ a₂ b₂, a₁ ≈ a₂ → b₁ ≈ b₂ → f a₁ b₁ = f a₂ b₂)
    [hf : ∀ a, DecidablePred (f a)]
    (q₁ : Quotient sa) : DecidablePred (Quotient.lift₂ f h q₁) :=
  fun q₂ ↦ Quotient.recOnSubsingleton₂ q₁ q₂ hf


instance (q : Quotient sa) (f : α → Prop) (h : ∀ a b, a ≈ b → f a = f b) [DecidablePred f] :
    Decidable (Quotient.liftOn q f h) :=
  Quotient.lift.decidablePred _ _ _


instance (q₁ : Quotient sa) (q₂ : Quotient sb) (f : α → β → Prop)
    (h : ∀ a₁ b₁ a₂ b₂, a₁ ≈ a₂ → b₁ ≈ b₂ → f a₁ b₁ = f a₂ b₂) [∀ a, DecidablePred (f a)] :
    Decidable (Quotient.liftOn₂ q₁ q₂ f h) :=
  Quotient.lift₂.decidablePred _ _ _ _


theorem Quot.eq {α : Type*} {r : α → α → Prop} {x y : α} :
    Quot.mk r x = Quot.mk r y ↔ Relation.EqvGen r x y :=
  ⟨Quot.eqvGen_exact, Quot.eqvGen_sound⟩


@[simp]
theorem Quotient.eq {r : Setoid α} {x y : α} : Quotient.mk r x = ⟦y⟧ ↔ r x y :=
  ⟨Quotient.exact, Quotient.sound⟩


theorem Quotient.eq_iff_equiv {r : Setoid α} {x y : α} : Quotient.mk r x = ⟦y⟧ ↔ x ≈ y :=
  Quotient.eq


theorem Quotient.forall {α : Sort*} {s : Setoid α} {p : Quotient s → Prop} :
    (∀ a, p a) ↔ ∀ a : α, p ⟦a⟧ :=
  ⟨fun h _ ↦ h _, fun h a ↦ a.ind h⟩


theorem Quotient.exists {α : Sort*} {s : Setoid α} {p : Quotient s → Prop} :
    (∃ a, p a) ↔ ∃ a : α, p ⟦a⟧ :=
  ⟨fun ⟨q, hq⟩ ↦ q.ind (motive := (p · → _)) .intro hq, fun ⟨a, ha⟩ ↦ ⟨⟦a⟧, ha⟩⟩


@[simp]
theorem Quotient.lift_mk {s : Setoid α} (f : α → β) (h : ∀ a b : α, a ≈ b → f a = f b) (x : α) :
    Quotient.lift f h (Quotient.mk s x) = f x :=
  rfl


@[simp]
theorem Quotient.lift_comp_mk {_ : Setoid α} (f : α → β) (h : ∀ a b : α, a ≈ b → f a = f b) :
    Quotient.lift f h ∘ Quotient.mk _ = f :=
  rfl


@[simp]
theorem Quotient.lift₂_mk {α : Sort*} {β : Sort*} {γ : Sort*} {_ : Setoid α} {_ : Setoid β}
    (f : α → β → γ)
    (h : ∀ (a₁ : α) (a₂ : β) (b₁ : α) (b₂ : β), a₁ ≈ b₁ → a₂ ≈ b₂ → f a₁ a₂ = f b₁ b₂)
    (a : α) (b : β) :
    Quotient.lift₂ f h (Quotient.mk _ a) (Quotient.mk _ b) = f a b :=
  rfl


theorem Quotient.liftOn_mk {s : Setoid α} (f : α → β) (h : ∀ a b : α, a ≈ b → f a = f b) (x : α) :
    Quotient.liftOn (Quotient.mk s x) f h = f x :=
  rfl


@[simp]
theorem Quotient.liftOn₂_mk {α : Sort*} {β : Sort*} {_ : Setoid α} (f : α → α → β)
    (h : ∀ a₁ a₂ b₁ b₂ : α, a₁ ≈ b₁ → a₂ ≈ b₂ → f a₁ a₂ = f b₁ b₂) (x y : α) :
    Quotient.liftOn₂ (Quotient.mk _ x) (Quotient.mk _ y) f h = f x y :=
  rfl


/-- `Quot.mk r` is a surjective function. -/
theorem Quot.mk_surjective {r : α → α → Prop} : Function.Surjective (Quot.mk r) :=
  Quot.exists_rep


/-- `Quotient.mk` is a surjective function. -/
theorem Quotient.mk_surjective {s : Setoid α} :
    Function.Surjective (Quotient.mk s) :=
  Quot.exists_rep


/-- `Quotient.mk'` is a surjective function. -/
theorem Quotient.mk'_surjective [s : Setoid α] :
    Function.Surjective (Quotient.mk' : α → Quotient s) :=
  Quot.exists_rep


/-- `Quot.mk r` is a surjective function. -/
@[deprecated Quot.mk_surjective (since := "2024-09-02")]
theorem surjective_quot_mk (r : α → α → Prop) : Function.Surjective (Quot.mk r) :=
  Quot.exists_rep


/-- `Quotient.mk` is a surjective function. -/
@[deprecated Quotient.mk_surjective (since := "2024-09-02")]
theorem surjective_quotient_mk {α : Sort*} (s : Setoid α) :
    Function.Surjective (Quotient.mk s) :=
  Quot.exists_rep


/-- `Quotient.mk'` is a surjective function. -/
@[deprecated Quotient.mk'_surjective (since := "2024-09-02")]
theorem surjective_quotient_mk' (α : Sort*) [s : Setoid α] :
    Function.Surjective (Quotient.mk' : α → Quotient s) :=
  Quot.exists_rep


/-- Choose an element of the equivalence class using the axiom of choice.
  Sound but noncomputable. -/
noncomputable def Quot.out {r : α → α → Prop} (q : Quot r) : α :=
  Classical.choose (Quot.exists_rep q)


/-- Unwrap the VM representation of a quotient to obtain an element of the equivalence class.
  Computable but unsound. -/
unsafe def Quot.unquot {r : α → α → Prop} : Quot r → α :=
  cast lcProof -- Porting note: was `unchecked_cast` before, which unfolds to `cast undefined`


@[simp]
theorem Quot.out_eq {r : α → α → Prop} (q : Quot r) : Quot.mk r q.out = q :=
  Classical.choose_spec (Quot.exists_rep q)


/-- Choose an element of the equivalence class using the axiom of choice.
  Sound but noncomputable. -/
noncomputable def Quotient.out {s : Setoid α} : Quotient s → α :=
  Quot.out


@[simp]
theorem Quotient.out_eq {s : Setoid α} (q : Quotient s) : ⟦q.out⟧ = q :=
  Quot.out_eq q


theorem Quotient.mk_out {s : Setoid α} (a : α) : s (⟦a⟧ : Quotient s).out a :=
  Quotient.exact (Quotient.out_eq _)


theorem Quotient.mk_eq_iff_out {s : Setoid α} {x : α} {y : Quotient s} :
    ⟦x⟧ = y ↔ x ≈ Quotient.out y := by
  /-
    α : Sort u_1
    s : Setoid α
    x : α
    y : Quotient s
    ⊢ Iff (Eq (Quotient.mk s x) y) (HasEquiv.Equiv x y.out)
  -/
  refine Iff.trans ?_ Quotient.eq
  /-
    α : Sort u_1
    s : Setoid α
    x : α
    y : Quotient s
    ⊢ Iff (Eq (Quotient.mk s x) y) (Eq (Quotient.mk s x) (Quotient.mk s y.out))
  -/
  rw [Quotient.out_eq y]
  /-
    🎉 no goals
  -/


theorem Quotient.eq_mk_iff_out {s : Setoid α} {x : Quotient s} {y : α} :
    x = ⟦y⟧ ↔ Quotient.out x ≈ y := by
  /-
    α : Sort u_1
    s : Setoid α
    x : Quotient s
    y : α
    ⊢ Iff (Eq x (Quotient.mk s y)) (HasEquiv.Equiv x.out y)
  -/
  refine Iff.trans ?_ Quotient.eq
  /-
    α : Sort u_1
    s : Setoid α
    x : Quotient s
    y : α
    ⊢ Iff (Eq x (Quotient.mk s y)) (Eq (Quotient.mk s x.out) (Quotient.mk s y))
  -/
  rw [Quotient.out_eq x]
  /-
    🎉 no goals
  -/


@[simp]
theorem Quotient.out_equiv_out {s : Setoid α} {x y : Quotient s} : x.out ≈ y.out ↔ x = y := by
  /-
    α : Sort u_1
    s : Setoid α
    x y : Quotient s
    ⊢ Iff (HasEquiv.Equiv x.out y.out) (Eq x y)
  -/
  rw [← Quotient.eq_mk_iff_out, Quotient.out_eq]
  /-
    🎉 no goals
  -/


theorem Quotient.out_injective {s : Setoid α} : Function.Injective (@Quotient.out α s) :=
  fun _ _ h ↦ Quotient.out_equiv_out.1 <| h ▸ Setoid.refl _


@[simp]
theorem Quotient.out_inj {s : Setoid α} {x y : Quotient s} : x.out = y.out ↔ x = y :=
  ⟨fun h ↦ Quotient.out_injective h, fun h ↦ h ▸ rfl⟩


instance piSetoid {ι : Sort*} {α : ι → Sort*} [∀ i, Setoid (α i)] : Setoid (∀ i, α i) where
  r a b := ∀ i, a i ≈ b i
  iseqv := ⟨fun _ _ ↦ Setoid.refl _,
            fun h _ ↦ Setoid.symm (h _),
            fun h₁ h₂ _ ↦ Setoid.trans (h₁ _) (h₂ _)⟩


/-- Given a class of functions `q : @Quotient (∀ i, α i) _`, returns the class of `i`-th projection
`Quotient (S i)`. -/
def Quotient.eval {ι : Type*} {α : ι → Sort*} {S : ∀ i, Setoid (α i)}
                                  /-
                                    α✝ : Sort u_1
                                    β : Sort u_2
                                    ι : Type u_3
                                    α : ι → Sort u_4
                                    S : (i : ι) → Setoid (α i)
                                    ⊢ Setoid ((i : ι) → α i)
                                  -/
    (q : @Quotient (∀ i, α i) (by infer_instance)) (i : ι) : Quotient (S i) :=
                                  /-
                                    🎉 no goals
                                  -/
                             /-
                               α✝ : Sort u_1
                               β : Sort u_2
                               ι : Type u_3
                               α : ι → Sort u_4
                               S : (i : ι) → Setoid (α i)
                               q : Quotient inferInstance
                               i : ι
                               x✝¹ x✝ : (i : ι) → α i
                               h : HasEquiv.Equiv x✝¹ x✝
                               ⊢ HasEquiv.Equiv ((fun x => x i) x✝¹) ((fun x => x i) x✝)
                             -/
  q.map (· i) fun _ _ h ↦ by exact h i
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem Quotient.eval_mk {ι : Type*} {α : ι → Type*} {S : ∀ i, Setoid (α i)} (f : ∀ i, α i) :
    Quotient.eval (S := S) ⟦f⟧ = fun i ↦ ⟦f i⟧ :=
  rfl


/-- Given a function `f : Π i, Quotient (S i)`, returns the class of functions `Π i, α i` sending
each `i` to an element of the class `f i`. -/
noncomputable def Quotient.choice {ι : Type*} {α : ι → Type*} {S : ∀ i, Setoid (α i)}
    (f : ∀ i, Quotient (S i)) :
                             /-
                               α✝ : Sort u_1
                               β : Sort u_2
                               ι : Type u_3
                               α : ι → Type u_4
                               S : (i : ι) → Setoid (α i)
                               f : (i : ι) → Quotient (S i)
                               ⊢ Setoid ((i : ι) → α i)
                             -/
    @Quotient (∀ i, α i) (by infer_instance) :=
                             /-
                               🎉 no goals
                             -/
  ⟦fun i ↦ (f i).out⟧


@[simp]
theorem Quotient.choice_eq {ι : Type*} {α : ι → Type*} {S : ∀ i, Setoid (α i)} (f : ∀ i, α i) :
    (Quotient.choice (S := S) fun i ↦ ⟦f i⟧) = ⟦f⟧ :=
  Quotient.sound fun _ ↦ Quotient.mk_out _


@[elab_as_elim]
theorem Quotient.induction_on_pi {ι : Type*} {α : ι → Sort*} {s : ∀ i, Setoid (α i)}
    {p : (∀ i, Quotient (s i)) → Prop} (f : ∀ i, Quotient (s i))
    (h : ∀ a : ∀ i, α i, p fun i ↦ ⟦a i⟧) : p f := by
  /-
    ι : Type u_3
    α : ι → Sort u_4
    s : (i : ι) → Setoid (α i)
    p : ((i : ι) → Quotient (s i)) → Prop
    f : (i : ι) → Quotient (s i)
    h : ∀ (a : (i : ι) → α i), p fun i => Quotient.mk (s i) (a i)
    ⊢ p f
  -/
  rw [← (funext fun i ↦ Quotient.out_eq (f i) : (fun i ↦ ⟦(f i).out⟧) = f)]
  /-
    ι : Type u_3
    α : ι → Sort u_4
    s : (i : ι) → Setoid (α i)
    p : ((i : ι) → Quotient (s i)) → Prop
    f : (i : ι) → Quotient (s i)
    h : ∀ (a : (i : ι) → α i), p fun i => Quotient.mk (s i) (a i)
    ⊢ p fun i => Quotient.mk (s i) (f i).out
  -/
  apply h
  /-
    🎉 no goals
  -/


theorem nonempty_quotient_iff (s : Setoid α) : Nonempty (Quotient s) ↔ Nonempty α :=
  ⟨fun ⟨a⟩ ↦ Quotient.inductionOn a Nonempty.intro, fun ⟨a⟩ ↦ ⟨⟦a⟧⟩⟩


theorem true_equivalence : @Equivalence α fun _ _ ↦ True :=
  ⟨fun _ ↦ trivial, fun _ ↦ trivial, fun _ _ ↦ trivial⟩


/-- Always-true relation as a `Setoid`.

Note that in later files the preferred spelling is `⊤ : Setoid α`. -/
def trueSetoid : Setoid α :=
  ⟨_, true_equivalence⟩


/-- `Trunc α` is the quotient of `α` by the always-true relation. This
  is related to the propositional truncation in HoTT, and is similar
  in effect to `Nonempty α`, but unlike `Nonempty α`, `Trunc α` is data,
  so the VM representation is the same as `α`, and so this can be used to
  maintain computability. -/
def Trunc.{u} (α : Sort u) : Sort u :=
  @Quotient α trueSetoid


/-- Constructor for `Trunc α` -/
def mk (a : α) : Trunc α :=
  Quot.mk _ a


instance [Inhabited α] : Inhabited (Trunc α) :=
  ⟨mk default⟩


/-- Any constant function lifts to a function out of the truncation -/
def lift (f : α → β) (c : ∀ a b : α, f a = f b) : Trunc α → β :=
  Quot.lift f fun a b _ ↦ c a b


theorem ind {β : Trunc α → Prop} : (∀ a : α, β (mk a)) → ∀ q : Trunc α, β q :=
  Quot.ind


protected theorem lift_mk (f : α → β) (c) (a : α) : lift f c (mk a) = f a :=
  rfl


/-- Lift a constant function on `q : Trunc α`. -/
protected def liftOn (q : Trunc α) (f : α → β) (c : ∀ a b : α, f a = f b) : β :=
  lift f c q


@[elab_as_elim]
protected theorem induction_on {β : Trunc α → Prop} (q : Trunc α) (h : ∀ a, β (mk a)) : β q :=
  ind h q


theorem exists_rep (q : Trunc α) : ∃ a : α, mk a = q :=
  Quot.exists_rep q


@[elab_as_elim]
protected theorem induction_on₂ {C : Trunc α → Trunc β → Prop} (q₁ : Trunc α) (q₂ : Trunc β)
    (h : ∀ a b, C (mk a) (mk b)) : C q₁ q₂ :=
  Trunc.induction_on q₁ fun a₁ ↦ Trunc.induction_on q₂ (h a₁)


protected theorem eq (a b : Trunc α) : a = b :=
  Trunc.induction_on₂ a b fun _ _ ↦ Quot.sound trivial


instance instSubsingletonTrunc : Subsingleton (Trunc α) :=
  ⟨Trunc.eq⟩


/-- The `bind` operator for the `Trunc` monad. -/
def bind (q : Trunc α) (f : α → Trunc β) : Trunc β :=
  Trunc.liftOn q f fun _ _ ↦ Trunc.eq _ _


/-- A function `f : α → β` defines a function `map f : Trunc α → Trunc β`. -/
def map (f : α → β) (q : Trunc α) : Trunc β :=
  bind q (Trunc.mk ∘ f)


instance : Monad Trunc where
  pure := @Trunc.mk
  bind := @Trunc.bind


instance : LawfulMonad Trunc where
  id_map _ := Trunc.eq _ _
  pure_bind _ _ := rfl
  bind_assoc _ _ _ := Trunc.eq _ _
  -- Porting note: the fields below are new in Lean 4
  map_const := rfl
  seqLeft_eq _ _ := Trunc.eq _ _
  seqRight_eq _ _ := Trunc.eq _ _
  pure_seq _ _ := rfl
  bind_pure_comp _ _ := rfl
  bind_map _ _ := rfl


/-- Recursion/induction principle for `Trunc`. -/
@[elab_as_elim]
protected def rec (f : ∀ a, C (mk a))
    (h : ∀ a b : α, (Eq.ndrec (f a) (Trunc.eq (mk a) (mk b)) : C (mk b)) = f b)
    (q : Trunc α) : C q :=
  Quot.rec f (fun a b _ ↦ h a b) q


/-- A version of `Trunc.rec` taking `q : Trunc α` as the first argument. -/
@[elab_as_elim]
protected def recOn (q : Trunc α) (f : ∀ a, C (mk a))
    (h : ∀ a b : α, (Eq.ndrec (f a) (Trunc.eq (mk a) (mk b)) : C (mk b)) = f b) : C q :=
  Trunc.rec f h q


/-- A version of `Trunc.recOn` assuming the codomain is a `Subsingleton`. -/
@[elab_as_elim]
protected def recOnSubsingleton [∀ a, Subsingleton (C (mk a))] (q : Trunc α) (f : ∀ a, C (mk a)) :
    C q :=
  Trunc.rec f (fun _ b ↦ Subsingleton.elim _ (f b)) q


/-- Noncomputably extract a representative of `Trunc α` (using the axiom of choice). -/
noncomputable def out : Trunc α → α :=
  Quot.out


@[simp]
theorem out_eq (q : Trunc α) : mk q.out = q :=
  Trunc.eq _ _


protected theorem nonempty (q : Trunc α) : Nonempty α :=
  nonempty_of_exists q.exists_rep


/-- A version of `Quotient.mk` taking `{s : Setoid α}` as an implicit argument instead of an
instance argument. -/
protected abbrev mk'' (a : α) : Quotient s₁ :=
  ⟦a⟧


/-- `Quotient.mk''` is a surjective function. -/
theorem mk''_surjective : Function.Surjective (Quotient.mk'' : α → Quotient s₁) :=
  Quot.exists_rep


@[deprecated (since := "2024-09-02")]
alias surjective_Quotient_mk'' := mk''_surjective


/-- A version of `Quotient.liftOn` taking `{s : Setoid α}` as an implicit argument instead of an
instance argument. -/
protected def liftOn' (q : Quotient s₁) (f : α → φ) (h : ∀ a b, s₁ a b → f a = f b) :
    φ :=
  Quotient.liftOn q f h


@[simp]
protected theorem liftOn'_mk'' (f : α → φ) (h) (x : α) :
    Quotient.liftOn' (@Quotient.mk'' _ s₁ x) f h = f x :=
  rfl


@[simp] lemma surjective_liftOn' {f : α → φ} (h) :
    Function.Surjective (fun x : Quotient s₁ ↦ x.liftOn' f h) ↔ Function.Surjective f :=
  Quot.surjective_lift _


/-- A version of `Quotient.liftOn₂` taking `{s₁ : Setoid α} {s₂ : Setoid β}` as implicit arguments
instead of instance arguments. -/
protected def liftOn₂' (q₁ : Quotient s₁) (q₂ : Quotient s₂) (f : α → β → γ)
    (h : ∀ a₁ a₂ b₁ b₂, s₁ a₁ b₁ → s₂ a₂ b₂ → f a₁ a₂ = f b₁ b₂) : γ :=
  Quotient.liftOn₂ q₁ q₂ f h


@[simp]
protected theorem liftOn₂'_mk'' (f : α → β → γ) (h) (a : α) (b : β) :
    Quotient.liftOn₂' (@Quotient.mk'' _ s₁ a) (@Quotient.mk'' _ s₂ b) f h = f a b :=
  rfl


/-- A version of `Quotient.ind` taking `{s : Setoid α}` as an implicit argument instead of an
instance argument. -/
@[elab_as_elim]
protected theorem ind' {p : Quotient s₁ → Prop} (h : ∀ a, p (Quotient.mk'' a)) (q : Quotient s₁) :
    p q :=
  Quotient.ind h q


/-- A version of `Quotient.ind₂` taking `{s₁ : Setoid α} {s₂ : Setoid β}` as implicit arguments
instead of instance arguments. -/
@[elab_as_elim]
protected theorem ind₂' {p : Quotient s₁ → Quotient s₂ → Prop}
    (h : ∀ a₁ a₂, p (Quotient.mk'' a₁) (Quotient.mk'' a₂))
    (q₁ : Quotient s₁) (q₂ : Quotient s₂) : p q₁ q₂ :=
  Quotient.ind₂ h q₁ q₂


/-- A version of `Quotient.inductionOn` taking `{s : Setoid α}` as an implicit argument instead
of an instance argument. -/
@[elab_as_elim]
protected theorem inductionOn' {p : Quotient s₁ → Prop} (q : Quotient s₁)
    (h : ∀ a, p (Quotient.mk'' a)) : p q :=
  Quotient.inductionOn q h


/-- A version of `Quotient.inductionOn₂` taking `{s₁ : Setoid α} {s₂ : Setoid β}` as implicit
arguments instead of instance arguments. -/
@[elab_as_elim]
protected theorem inductionOn₂' {p : Quotient s₁ → Quotient s₂ → Prop} (q₁ : Quotient s₁)
    (q₂ : Quotient s₂)
    (h : ∀ a₁ a₂, p (Quotient.mk'' a₁) (Quotient.mk'' a₂)) : p q₁ q₂ :=
  Quotient.inductionOn₂ q₁ q₂ h


/-- A version of `Quotient.inductionOn₃` taking `{s₁ : Setoid α} {s₂ : Setoid β} {s₃ : Setoid γ}`
as implicit arguments instead of instance arguments. -/
@[elab_as_elim]
protected theorem inductionOn₃' {p : Quotient s₁ → Quotient s₂ → Quotient s₃ → Prop}
    (q₁ : Quotient s₁) (q₂ : Quotient s₂) (q₃ : Quotient s₃)
    (h : ∀ a₁ a₂ a₃, p (Quotient.mk'' a₁) (Quotient.mk'' a₂) (Quotient.mk'' a₃)) :
    p q₁ q₂ q₃ :=
  Quotient.inductionOn₃ q₁ q₂ q₃ h


/-- A version of `Quotient.recOnSubsingleton` taking `{s₁ : Setoid α}` as an implicit argument
instead of an instance argument. -/
@[elab_as_elim]
protected def recOnSubsingleton' {φ : Quotient s₁ → Sort*} [∀ a, Subsingleton (φ ⟦a⟧)]
    (q : Quotient s₁)
    (f : ∀ a, φ (Quotient.mk'' a)) : φ q :=
  Quotient.recOnSubsingleton q f


/-- A version of `Quotient.recOnSubsingleton₂` taking `{s₁ : Setoid α} {s₂ : Setoid α}`
as implicit arguments instead of instance arguments. -/
@[elab_as_elim]
protected def recOnSubsingleton₂' {φ : Quotient s₁ → Quotient s₂ → Sort*}
    [∀ a b, Subsingleton (φ ⟦a⟧ ⟦b⟧)]
    (q₁ : Quotient s₁) (q₂ : Quotient s₂) (f : ∀ a₁ a₂, φ (Quotient.mk'' a₁) (Quotient.mk'' a₂)) :
    φ q₁ q₂ :=
  Quotient.recOnSubsingleton₂ q₁ q₂ f


/-- Recursion on a `Quotient` argument `a`, result type depends on `⟦a⟧`. -/
protected def hrecOn' {φ : Quotient s₁ → Sort*} (qa : Quotient s₁) (f : ∀ a, φ (Quotient.mk'' a))
    (c : ∀ a₁ a₂, a₁ ≈ a₂ → HEq (f a₁) (f a₂)) : φ qa :=
  Quot.hrecOn qa f c


@[simp]
theorem hrecOn'_mk'' {φ : Quotient s₁ → Sort*} (f : ∀ a, φ (Quotient.mk'' a))
    (c : ∀ a₁ a₂, a₁ ≈ a₂ → HEq (f a₁) (f a₂))
    (x : α) : (Quotient.mk'' x).hrecOn' f c = f x :=
  rfl


/-- Recursion on two `Quotient` arguments `a` and `b`, result type depends on `⟦a⟧` and `⟦b⟧`. -/
protected def hrecOn₂' {φ : Quotient s₁ → Quotient s₂ → Sort*} (qa : Quotient s₁)
    (qb : Quotient s₂) (f : ∀ a b, φ (Quotient.mk'' a) (Quotient.mk'' b))
    (c : ∀ a₁ b₁ a₂ b₂, a₁ ≈ a₂ → b₁ ≈ b₂ → HEq (f a₁ b₁) (f a₂ b₂)) :
    φ qa qb :=
  Quotient.hrecOn₂ qa qb f c


@[simp]
theorem hrecOn₂'_mk'' {φ : Quotient s₁ → Quotient s₂ → Sort*}
    (f : ∀ a b, φ (Quotient.mk'' a) (Quotient.mk'' b))
    (c : ∀ a₁ b₁ a₂ b₂, a₁ ≈ a₂ → b₁ ≈ b₂ → HEq (f a₁ b₁) (f a₂ b₂)) (x : α) (qb : Quotient s₂) :
    (Quotient.mk'' x).hrecOn₂' qb f c = qb.hrecOn' (f x) fun _ _ ↦ c _ _ _ _ (Setoid.refl _) :=
  rfl


/-- Map a function `f : α → β` that sends equivalent elements to equivalent elements
to a function `Quotient sa → Quotient sb`. Useful to define unary operations on quotients. -/
protected def map' (f : α → β) (h : ∀ a b, s₁.r a b → s₂.r (f a) (f b)) :
    Quotient s₁ → Quotient s₂ :=
  Quot.map f h


@[simp]
theorem map'_mk'' (f : α → β) (h) (x : α) :
    (Quotient.mk'' x : Quotient s₁).map' f h = (Quotient.mk'' (f x) : Quotient s₂) :=
  rfl


/-- A version of `Quotient.map₂` using curly braces and unification. -/
@[deprecated (since := "2024-12-01")] protected alias map₂' := Quotient.map₂


theorem exact' {a b : α} :
    (Quotient.mk'' a : Quotient s₁) = Quotient.mk'' b → s₁ a b :=
  Quotient.exact


theorem sound' {a b : α} : s₁ a b → @Quotient.mk'' α s₁ a = Quotient.mk'' b :=
  Quotient.sound


@[simp]
protected theorem eq' {s₁ : Setoid α} {a b : α} :
    @Quotient.mk' α s₁ a = @Quotient.mk' α s₁ b ↔ s₁ a b :=
  Quotient.eq


protected theorem eq'' {a b : α} : @Quotient.mk'' α s₁ a = Quotient.mk'' b ↔ s₁ a b :=
  Quotient.eq


@[deprecated (since := "2024-10-19")] alias out' := out


theorem out_eq' (q : Quotient s₁) : Quotient.mk'' q.out = q :=
  q.out_eq


theorem mk_out' (a : α) : s₁ (Quotient.mk'' a : Quotient s₁).out a :=
  Quotient.exact (Quotient.out_eq _)


protected theorem mk''_eq_mk : Quotient.mk'' = Quotient.mk s :=
  rfl


@[simp]
protected theorem liftOn'_mk (x : α) (f : α → β) (h) : (Quotient.mk s x).liftOn' f h = f x :=
  rfl


@[simp]
protected theorem liftOn₂'_mk {t : Setoid β} (f : α → β → γ) (h) (a : α) (b : β) :
    Quotient.liftOn₂' (Quotient.mk s a) (Quotient.mk t b) f h = f a b :=
  Quotient.liftOn₂'_mk'' _ _ _ _


theorem map'_mk {t : Setoid β} (f : α → β) (h) (x : α) :
    (Quotient.mk s x).map' f h = (Quotient.mk t (f x)) :=
  rfl


instance (q : Quotient s₁) (f : α → Prop) (h : ∀ a b, s₁ a b → f a = f b)
    [DecidablePred f] :
    Decidable (Quotient.liftOn' q f h) :=
  Quotient.lift.decidablePred _ _ q


instance (q₁ : Quotient s₁) (q₂ : Quotient s₂) (f : α → β → Prop)
    (h : ∀ a₁ b₁ a₂ b₂, s₁ a₁ a₂ → s₂ b₁ b₂ → f a₁ b₁ = f a₂ b₂)
    [∀ a, DecidablePred (f a)] :
    Decidable (Quotient.liftOn₂' q₁ q₂ f h) :=
  Quotient.lift₂.decidablePred _ h _ _


@[simp]
lemma Equivalence.quot_mk_eq_iff {α : Type*} {r : α → α → Prop} (h : Equivalence r) (x y : α) :
    Quot.mk r x = Quot.mk r y ↔ r x y := by
  /-
    α : Type u_3
    r : α → α → Prop
    h : Equivalence r
    x y : α
    ⊢ Iff (Eq (Quot.mk r x) (Quot.mk r y)) (r x y)
  -/
  constructor
    /-
      case mp
      α : Type u_3
      r : α → α → Prop
      h : Equivalence r
      x y : α
      ⊢ Eq (Quot.mk r x) (Quot.mk r y) → r x y
    -/
  · rw [Quot.eq]
    /-
      case mp
      α : Type u_3
      r : α → α → Prop
      h : Equivalence r
      x y : α
      ⊢ Relation.EqvGen r x y → r x y
    -/
    intro hxy
    induction hxy with
    | rel _ _ H => exact H
    | refl _ => exact h.refl _
    | symm _ _ _ H => exact h.symm H
    | trans _ _ _ _ _ h₁₂ h₂₃ => exact h.trans h₁₂ h₂₃
    /-
      case mpr
      α : Type u_3
      r : α → α → Prop
      h : Equivalence r
      x y : α
      ⊢ r x y → Eq (Quot.mk r x) (Quot.mk r y)
    -/
  · exact Quot.sound
    /-
      🎉 no goals
    -/

