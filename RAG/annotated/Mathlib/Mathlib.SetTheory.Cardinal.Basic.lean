/-- The equivalence relation on types given by equivalence (bijective correspondence) of types.
  Quotienting by this equivalence relation gives the cardinal numbers.
-/
instance Cardinal.isEquivalent : Setoid (Type u) where
  r α β := Nonempty (α ≃ β)
  iseqv := ⟨
    fun α => ⟨Equiv.refl α⟩,
    fun ⟨e⟩ => ⟨e.symm⟩,
    fun ⟨e₁⟩ ⟨e₂⟩ => ⟨e₁.trans e₂⟩⟩


/-- `Cardinal.{u}` is the type of cardinal numbers in `Type u`,
  defined as the quotient of `Type u` by existence of an equivalence
  (a bijection with explicit inverse). -/
@[pp_with_univ]
def Cardinal : Type (u + 1) :=
  Quotient Cardinal.isEquivalent


/-- The cardinal number of a type -/
def mk : Type u → Cardinal :=
  Quotient.mk'


@[inherit_doc]
scoped prefix:max "#" => Cardinal.mk


instance canLiftCardinalType : CanLift Cardinal.{u} (Type u) mk fun _ => True :=
  ⟨fun c _ => Quot.inductionOn c fun α => ⟨α, rfl⟩⟩


@[elab_as_elim]
theorem inductionOn {p : Cardinal → Prop} (c : Cardinal) (h : ∀ α, p #α) : p c :=
  Quotient.inductionOn c h


@[elab_as_elim]
theorem inductionOn₂ {p : Cardinal → Cardinal → Prop} (c₁ : Cardinal) (c₂ : Cardinal)
    (h : ∀ α β, p #α #β) : p c₁ c₂ :=
  Quotient.inductionOn₂ c₁ c₂ h


@[elab_as_elim]
theorem inductionOn₃ {p : Cardinal → Cardinal → Cardinal → Prop} (c₁ : Cardinal) (c₂ : Cardinal)
    (c₃ : Cardinal) (h : ∀ α β γ, p #α #β #γ) : p c₁ c₂ c₃ :=
  Quotient.inductionOn₃ c₁ c₂ c₃ h


theorem induction_on_pi {ι : Type u} {p : (ι → Cardinal.{v}) → Prop}
    (f : ι → Cardinal.{v}) (h : ∀ f : ι → Type v, p fun i ↦ #(f i)) : p f :=
  Quotient.induction_on_pi f h


protected theorem eq : #α = #β ↔ Nonempty (α ≃ β) :=
  Quotient.eq'


/-- Avoid using `Quotient.mk` to construct a `Cardinal` directly -/
@[deprecated "No deprecation message was provided." (since := "2024-10-24")]
theorem mk'_def (α : Type u) : @Eq Cardinal ⟦α⟧ #α :=
  rfl


@[simp]
theorem mk_out (c : Cardinal) : #c.out = c :=
  Quotient.out_eq _


/-- The representative of the cardinal of a type is equivalent to the original type. -/
def outMkEquiv {α : Type v} : (#α).out ≃ α :=
                                      /-
                                        α✝ β : Type u
                                        α : Type v
                                        ⊢ Eq (Cardinal.mk (Quotient.out (Cardinal.mk α))) (Cardinal.mk α)
                                      -/
  Nonempty.some <| Cardinal.eq.mp (by simp)
                                      /-
                                        🎉 no goals
                                      -/


theorem mk_congr (e : α ≃ β) : #α = #β :=
  Quot.sound ⟨e⟩


alias _root_.Equiv.cardinal_eq := mk_congr


/-- Lift a function between `Type*`s to a function between `Cardinal`s. -/
def map (f : Type u → Type v) (hf : ∀ α β, α ≃ β → f α ≃ f β) : Cardinal.{u} → Cardinal.{v} :=
  Quotient.map f fun α β ⟨e⟩ => ⟨hf α β e⟩


@[simp]
theorem map_mk (f : Type u → Type v) (hf : ∀ α β, α ≃ β → f α ≃ f β) (α : Type u) :
    map f hf #α = #(f α) :=
  rfl


/-- Lift a binary operation `Type* → Type* → Type*` to a binary operation on `Cardinal`s. -/
def map₂ (f : Type u → Type v → Type w) (hf : ∀ α β γ δ, α ≃ β → γ ≃ δ → f α γ ≃ f β δ) :
    Cardinal.{u} → Cardinal.{v} → Cardinal.{w} :=
  Quotient.map₂ f fun α β ⟨e₁⟩ γ δ ⟨e₂⟩ => ⟨hf α β γ δ e₁ e₂⟩


/-- The universe lift operation on cardinals. You can specify the universes explicitly with
  `lift.{u v} : Cardinal.{v} → Cardinal.{max v u}` -/
@[pp_with_univ]
def lift (c : Cardinal.{v}) : Cardinal.{max v u} :=
  map ULift.{u, v} (fun _ _ e => Equiv.ulift.trans <| e.trans Equiv.ulift.symm) c


@[simp]
theorem mk_uLift (α) : #(ULift.{v, u} α) = lift.{v} #α :=
  rfl


/-- `lift.{max u v, u}` equals `lift.{v, u}`.

Unfortunately, the simp lemma doesn't work. -/
theorem lift_umax : lift.{max u v, u} = lift.{v, u} :=
  funext fun a => inductionOn a fun _ => (Equiv.ulift.trans Equiv.ulift.symm).cardinal_eq


/-- `lift.{max v u, u}` equals `lift.{v, u}`. -/
@[deprecated lift_umax (since := "2024-10-24")]
theorem lift_umax' : lift.{max v u, u} = lift.{v, u} :=
  lift_umax


/-- A cardinal lifted to a lower or equal universe equals itself.

Unfortunately, the simp lemma doesn't work. -/
theorem lift_id' (a : Cardinal.{max u v}) : lift.{u} a = a :=
  inductionOn a fun _ => mk_congr Equiv.ulift


/-- A cardinal lifted to the same universe equals itself. -/
@[simp]
theorem lift_id (a : Cardinal) : lift.{u, u} a = a :=
  lift_id'.{u, u} a


/-- A cardinal lifted to the zero universe equals itself. -/
@[simp]
theorem lift_uzero (a : Cardinal.{u}) : lift.{0} a = a :=
  lift_id'.{0, u} a


@[simp]
theorem lift_lift.{u_1} (a : Cardinal.{u_1}) : lift.{w} (lift.{v} a) = lift.{max v w} a :=
  inductionOn a fun _ => (Equiv.ulift.trans <| Equiv.ulift.trans Equiv.ulift.symm).cardinal_eq


theorem out_lift_equiv (a : Cardinal.{u}) : Nonempty ((lift.{v} a).out ≃ a.out) := by
  /-
    a : Cardinal.{u}
    ⊢ Nonempty (Equiv (Quotient.out (Cardinal.lift.{v, u} a)) (Quotient.out a))
  -/
  rw [← mk_out a, ← mk_uLift, mk_out]
  /-
    a : Cardinal.{u}
    ⊢ Nonempty (Equiv (Quotient.out (Cardinal.mk (ULift.{v, u} (Quotient.out a)))) …
  -/
  exact ⟨outMkEquiv.trans Equiv.ulift⟩
  /-
    🎉 no goals
  -/


@[simp]
lemma mk_preimage_down {s : Set α} : #(ULift.down.{v} ⁻¹' s) = lift.{v} (#s) := by
  /-
    α : Type u
    s : Set α
    ⊢ Eq (Cardinal.mk ↑(Set.preimage ULift.down s)) (Cardinal.lift.{v, u} (Cardina …
  -/
  rw [← mk_uLift, Cardinal.eq]
  /-
    α : Type u
    s : Set α
    ⊢ Nonempty (Equiv (↑(Set.preimage ULift.down s)) (ULift.{v, u} ↑s))
  -/
  constructor
  /-
    case val
    α : Type u
    s : Set α
    ⊢ Equiv (↑(Set.preimage ULift.down s)) (ULift.{v, u} ↑s)
  -/
  let f : ULift.down ⁻¹' s → ULift s := fun x ↦ ULift.up (restrictPreimage s ULift.down x)
  have : Function.Bijective f :=
    ULift.up_bijective.comp (restrictPreimage_bijective _ (ULift.down_bijective))
  /-
    case val
    α : Type u
    s : Set α
    f : ↑(Set.preimage ULift.down s) → ULift.{?u.4289, u} ↑s := fun x => { down := …
    this : Function.Bijective f
    ⊢ Equiv (↑(Set.preimage ULift.down s)) (ULift.{v, u} ↑s)
  -/
  exact Equiv.ofBijective f this
  /-
    🎉 no goals
  -/


theorem lift_mk_eq {α : Type u} {β : Type v} :
    lift.{max v w} #α = lift.{max u w} #β ↔ Nonempty (α ≃ β) :=
  Quotient.eq'.trans
    ⟨fun ⟨f⟩ => ⟨Equiv.ulift.symm.trans <| f.trans Equiv.ulift⟩, fun ⟨f⟩ =>
      ⟨Equiv.ulift.trans <| f.trans Equiv.ulift.symm⟩⟩


/-- A variant of `Cardinal.lift_mk_eq` with specialized universes.
Because Lean often can not realize it should use this specialization itself,
we provide this statement separately so you don't have to solve the specialization problem either.
-/
theorem lift_mk_eq' {α : Type u} {β : Type v} : lift.{v} #α = lift.{u} #β ↔ Nonempty (α ≃ β) :=
  lift_mk_eq.{u, v, 0}

-- Porting note: simpNF is not happy with universe levels.

@[simp, nolint simpNF]
theorem lift_mk_shrink (α : Type u) [Small.{v} α] :
    Cardinal.lift.{max u w} #(Shrink.{v} α) = Cardinal.lift.{max v w} #α :=
  lift_mk_eq.2 ⟨(equivShrink α).symm⟩


@[simp]
theorem lift_mk_shrink' (α : Type u) [Small.{v} α] :
    Cardinal.lift.{u} #(Shrink.{v} α) = Cardinal.lift.{v} #α :=
  lift_mk_shrink.{u, v, 0} α


@[simp]
theorem lift_mk_shrink'' (α : Type max u v) [Small.{v} α] :
    Cardinal.lift.{u} #(Shrink.{v} α) = #α := by
  /-
    α : Type (max u v)
    inst✝ : Small.{v, max u v} α
    ⊢ Eq (Cardinal.lift.{u, v} (Cardinal.mk (Shrink.{v, max u v} α))) (Cardinal.mk …
  -/
  rw [← lift_umax, lift_mk_shrink.{max u v, v, 0} α, ← lift_umax, lift_id]
  /-
    🎉 no goals
  -/


/-- We define the order on cardinal numbers by `#α ≤ #β` if and only if
  there exists an embedding (injective function) from α to β. -/
instance : LE Cardinal.{u} :=
  ⟨fun q₁ q₂ =>
    Quotient.liftOn₂ q₁ q₂ (fun α β => Nonempty <| α ↪ β) fun _ _ _ _ ⟨e₁⟩ ⟨e₂⟩ =>
      propext ⟨fun ⟨e⟩ => ⟨e.congr e₁ e₂⟩, fun ⟨e⟩ => ⟨e.congr e₁.symm e₂.symm⟩⟩⟩


instance partialOrder : PartialOrder Cardinal.{u} where
  le := (· ≤ ·)
  le_refl := by
    /-
      α β : Type u
      ⊢ ∀ (a : Cardinal.{u}), LE.le a a
    -/
    rintro ⟨α⟩
    /-
      case mk
      α✝ β : Type u
      a✝ : Cardinal.{u}
      α : Type u
      ⊢ LE.le (Quot.mk (⇑Cardinal.isEquivalent) α) (Quot.mk (⇑Cardinal.isEquivalent) …
    -/
    exact ⟨Embedding.refl _⟩
    /-
      🎉 no goals
    -/
  le_trans := by
    /-
      α β : Type u
      ⊢ ∀ (a b c : Cardinal.{u}), LE.le a b → LE.le b c → LE.le a c
    -/
    rintro ⟨α⟩ ⟨β⟩ ⟨γ⟩ ⟨e₁⟩ ⟨e₂⟩
    /-
      case mk.mk.mk.intro.intro
      α✝ β✝ : Type u
      a✝ : Cardinal.{u}
      α : Type u
      b✝ : Cardinal.{u}
      β : Type u
      c✝ : Cardinal.{u}
      γ : Type u
      e₁ : Function.Embedding α β
      e₂ : Function.Embedding β γ
      ⊢ LE.le (Quot.mk (⇑Cardinal.isEquivalent) α) (Quot.mk (⇑Cardinal.isEquivalent) …
    -/
    exact ⟨e₁.trans e₂⟩
    /-
      🎉 no goals
    -/
  le_antisymm := by
    /-
      α β : Type u
      ⊢ ∀ (a b : Cardinal.{u}), LE.le a b → LE.le b a → Eq a b
    -/
    rintro ⟨α⟩ ⟨β⟩ ⟨e₁⟩ ⟨e₂⟩
    /-
      case mk.mk.intro.intro
      α✝ β✝ : Type u
      a✝ : Cardinal.{u}
      α : Type u
      b✝ : Cardinal.{u}
      β : Type u
      e₁ : Function.Embedding α β
      e₂ : Function.Embedding β α
      ⊢ Eq (Quot.mk (⇑Cardinal.isEquivalent) α) (Quot.mk (⇑Cardinal.isEquivalent) β)
    -/
    exact Quotient.sound (e₁.antisymm e₂)
    /-
      🎉 no goals
    -/


instance linearOrder : LinearOrder Cardinal.{u} :=
  { Cardinal.partialOrder with
    le_total := by
      /-
        α β : Type u
        ⊢ ∀ (a b : Cardinal.{u}), Or (LE.le a b) (LE.le b a)
      -/
      rintro ⟨α⟩ ⟨β⟩
      /-
        case mk.mk
        α✝ β✝ : Type u
        a✝ : Cardinal.{u}
        α : Type u
        b✝ : Cardinal.{u}
        β : Type u
        ⊢ Or (LE.le (Quot.mk (⇑Cardinal.isEquivalent) α) (Quot.mk (⇑Cardinal.isEquival …
      -/
      apply Embedding.total
      /-
        🎉 no goals
      -/
    decidableLE := Classical.decRel _ }


theorem le_def (α β : Type u) : #α ≤ #β ↔ Nonempty (α ↪ β) :=
  Iff.rfl


theorem mk_le_of_injective {α β : Type u} {f : α → β} (hf : Injective f) : #α ≤ #β :=
  ⟨⟨f, hf⟩⟩


theorem _root_.Function.Embedding.cardinal_le {α β : Type u} (f : α ↪ β) : #α ≤ #β :=
  ⟨f⟩


theorem mk_le_of_surjective {α β : Type u} {f : α → β} (hf : Surjective f) : #β ≤ #α :=
  ⟨Embedding.ofSurjective f hf⟩


theorem le_mk_iff_exists_set {c : Cardinal} {α : Type u} : c ≤ #α ↔ ∃ p : Set α, #p = c :=
  ⟨inductionOn c fun _ ⟨⟨f, hf⟩⟩ => ⟨Set.range f, (Equiv.ofInjective f hf).cardinal_eq.symm⟩,
    fun ⟨_, e⟩ => e ▸ ⟨⟨Subtype.val, fun _ _ => Subtype.eq⟩⟩⟩


theorem mk_subtype_le {α : Type u} (p : α → Prop) : #(Subtype p) ≤ #α :=
  ⟨Embedding.subtype p⟩


theorem mk_set_le (s : Set α) : #s ≤ #α :=
  mk_subtype_le s


theorem out_embedding {c c' : Cardinal} : c ≤ c' ↔ Nonempty (c.out ↪ c'.out) := by
  /-
    c c' : Cardinal.{u_1}
    ⊢ Iff (LE.le c c') (Nonempty (Function.Embedding (Quotient.out c) (Quotient.ou …
  -/
  conv_lhs => rw [← Cardinal.mk_out c, ← Cardinal.mk_out c', le_def]
  /-
    🎉 no goals
  -/


theorem lift_mk_le {α : Type v} {β : Type w} :
    lift.{max u w} #α ≤ lift.{max u v} #β ↔ Nonempty (α ↪ β) :=
  ⟨fun ⟨f⟩ => ⟨Embedding.congr Equiv.ulift Equiv.ulift f⟩, fun ⟨f⟩ =>
    ⟨Embedding.congr Equiv.ulift.symm Equiv.ulift.symm f⟩⟩


/-- A variant of `Cardinal.lift_mk_le` with specialized universes.
Because Lean often can not realize it should use this specialization itself,
we provide this statement separately so you don't have to solve the specialization problem either.
-/
theorem lift_mk_le' {α : Type u} {β : Type v} : lift.{v} #α ≤ lift.{u} #β ↔ Nonempty (α ↪ β) :=
  lift_mk_le.{0}


/-- `Cardinal.lift` as an `InitialSeg`. -/
@[simps!]
def liftInitialSeg : Cardinal.{u} ≤i Cardinal.{max u v} := by
  /-
    α β : Type u
    ⊢ InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
  -/
  refine ⟨(OrderEmbedding.ofMapLEIff lift ?_).ltEmbedding, ?_⟩ <;> intro a b
    /-
      case refine_1
      α β : Type u
      a b : Cardinal.{u}
      ⊢ Iff (LE.le (Cardinal.lift.{v, u} a) (Cardinal.lift.{v, u} b)) (LE.le a b)
    -/
  · refine inductionOn₂ a b fun _ _ ↦ ?_
    /-
      case refine_1
      α β : Type u
      a b : Cardinal.{u}
      x✝¹ x✝ : Type u
      ⊢ Iff (LE.le (Cardinal.lift.{v, u} (Cardinal.mk x✝¹)) (Cardinal.lift.{v, u} (C …
    -/
    rw [← lift_umax, lift_mk_le.{v, u, u}, le_def]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α β : Type u
      a : Cardinal.{u}
      b : Cardinal.{max u v}
      ⊢ LT.lt b ((OrderEmbedding.ofMapLEIff Cardinal.lift.{v, u} ⋯).ltEmbedding a) → …
    -/
  · refine inductionOn₂ a b fun α β h ↦ ?_
    /-
      case refine_2
      α✝ β✝ : Type u
      a : Cardinal.{u}
      b : Cardinal.{max u v}
      α : Type u
      β : Type (max u v)
      h : LT.lt (Cardinal.mk β) ((OrderEmbedding.ofMapLEIff Cardinal.lift.{v, u} ⋯). …
      ⊢ Membership.mem (Set.range ⇑(OrderEmbedding.ofMapLEIff Cardinal.lift.{v, u} ⋯ …
    -/
    obtain ⟨e⟩ := h.le
    /-
      case refine_2.intro
      α✝ β✝ : Type u
      a : Cardinal.{u}
      b : Cardinal.{max u v}
      α : Type u
      β : Type (max u v)
      h : LT.lt (Cardinal.mk β) ((OrderEmbedding.ofMapLEIff Cardinal.lift.{v, u} ⋯). …
      e : Function.Embedding β (ULift.{v, u} α)
      ⊢ Membership.mem (Set.range ⇑(OrderEmbedding.ofMapLEIff Cardinal.lift.{v, u} ⋯ …
    -/
    replace e := e.congr (Equiv.refl β) Equiv.ulift
    /-
      case refine_2.intro
      α✝ β✝ : Type u
      a : Cardinal.{u}
      b : Cardinal.{max u v}
      α : Type u
      β : Type (max u v)
      h : LT.lt (Cardinal.mk β) ((OrderEmbedding.ofMapLEIff Cardinal.lift.{v, u} ⋯). …
      e : Function.Embedding β α
      ⊢ Membership.mem (Set.range ⇑(OrderEmbedding.ofMapLEIff Cardinal.lift.{v, u} ⋯ …
    -/
    refine ⟨#(range e), mk_congr (Equiv.ulift.trans <| Equiv.symm ?_)⟩
    /-
      case refine_2.intro
      α✝ β✝ : Type u
      a : Cardinal.{u}
      b : Cardinal.{max u v}
      α : Type u
      β : Type (max u v)
      h : LT.lt (Cardinal.mk β) ((OrderEmbedding.ofMapLEIff Cardinal.lift.{v, u} ⋯). …
      e : Function.Embedding β α
      ⊢ Equiv β ↑(Set.range ⇑e)
    -/
    apply (e.codRestrict _ mem_range_self).equivOfSurjective
    /-
      case refine_2.intro
      α✝ β✝ : Type u
      a : Cardinal.{u}
      b : Cardinal.{max u v}
      α : Type u
      β : Type (max u v)
      h : LT.lt (Cardinal.mk β) ((OrderEmbedding.ofMapLEIff Cardinal.lift.{v, u} ⋯). …
      e : Function.Embedding β α
      ⊢ Function.Surjective ⇑(Function.Embedding.codRestrict (Set.range ⇑e) e ⋯)
    -/
    rintro ⟨a, ⟨b, rfl⟩⟩
    /-
      case refine_2.intro.mk.intro
      α✝ β✝ : Type u
      a : Cardinal.{u}
      b✝ : Cardinal.{max u v}
      α : Type u
      β : Type (max u v)
      h : LT.lt (Cardinal.mk β) ((OrderEmbedding.ofMapLEIff Cardinal.lift.{v, u} ⋯). …
      e : Function.Embedding β α
      b : β
      ⊢ Exists fun a => Eq ((Function.Embedding.codRestrict (Set.range ⇑e) e ⋯) a) ⟨ …
    -/
    exact ⟨b, rfl⟩
    /-
      🎉 no goals
    -/


theorem mem_range_lift_of_le {a : Cardinal.{u}} {b : Cardinal.{max u v}} :
    b ≤ lift.{v, u} a → b ∈ Set.range lift.{v, u} :=
  liftInitialSeg.mem_range_of_le


@[deprecated mem_range_lift_of_le (since := "2024-10-07")]
theorem lift_down {a : Cardinal.{u}} {b : Cardinal.{max u v}} :
    b ≤ lift.{v, u} a → ∃ a', lift.{v, u} a' = b :=
  mem_range_lift_of_le


/-- `Cardinal.lift` as an `OrderEmbedding`. -/
@[deprecated Cardinal.liftInitialSeg (since := "2024-10-07")]
def liftOrderEmbedding : Cardinal.{v} ↪o Cardinal.{max v u} :=
  liftInitialSeg.toOrderEmbedding


theorem lift_injective : Injective lift.{u, v} :=
  liftInitialSeg.injective


@[simp]
theorem lift_inj {a b : Cardinal.{u}} : lift.{v, u} a = lift.{v, u} b ↔ a = b :=
  lift_injective.eq_iff


@[simp]
theorem lift_le {a b : Cardinal.{v}} : lift.{u} a ≤ lift.{u} b ↔ a ≤ b :=
  liftInitialSeg.le_iff_le


@[simp]
theorem lift_lt {a b : Cardinal.{u}} : lift.{v, u} a < lift.{v, u} b ↔ a < b :=
  liftInitialSeg.lt_iff_lt


theorem lift_strictMono : StrictMono lift := fun _ _ => lift_lt.2


theorem lift_monotone : Monotone lift :=
  lift_strictMono.monotone


@[simp]
theorem lift_min {a b : Cardinal} : lift.{u, v} (min a b) = min (lift.{u, v} a) (lift.{u, v} b) :=
  lift_monotone.map_min


@[simp]
theorem lift_max {a b : Cardinal} : lift.{u, v} (max a b) = max (lift.{u, v} a) (lift.{u, v} b) :=
  lift_monotone.map_max

-- Porting note: simpNF is not happy with universe levels.

@[simp, nolint simpNF]
theorem lift_umax_eq {a : Cardinal.{u}} {b : Cardinal.{v}} :
    lift.{max v w} a = lift.{max u w} b ↔ lift.{v} a = lift.{u} b := by
  /-
    a : Cardinal.{u}
    b : Cardinal.{v}
    ⊢ Iff (Eq (Cardinal.lift.{max v w, u} a) (Cardinal.lift.{max u w, v} b)) (Eq ( …
  -/
  rw [← lift_lift.{v, w, u}, ← lift_lift.{u, w, v}, lift_inj]
  /-
    🎉 no goals
  -/


theorem le_lift_iff {a : Cardinal.{u}} {b : Cardinal.{max u v}} :
    b ≤ lift.{v, u} a ↔ ∃ a' ≤ a, lift.{v, u} a' = b :=
  liftInitialSeg.le_apply_iff


theorem lt_lift_iff {a : Cardinal.{u}} {b : Cardinal.{max u v}} :
    b < lift.{v, u} a ↔ ∃ a' < a, lift.{v, u} a' = b :=
  liftInitialSeg.lt_apply_iff


instance : Zero Cardinal.{u} :=
  -- `PEmpty` might be more canonical, but this is convenient for defeq with natCast
  ⟨lift #(Fin 0)⟩


instance : Inhabited Cardinal.{u} :=
  ⟨0⟩


@[simp]
theorem mk_eq_zero (α : Type u) [IsEmpty α] : #α = 0 :=
  (Equiv.equivOfIsEmpty α (ULift (Fin 0))).cardinal_eq


@[simp]
theorem lift_zero : lift 0 = 0 := mk_eq_zero _


@[simp]
theorem lift_eq_zero {a : Cardinal.{v}} : lift.{u} a = 0 ↔ a = 0 :=
  lift_injective.eq_iff' lift_zero


theorem mk_eq_zero_iff {α : Type u} : #α = 0 ↔ IsEmpty α :=
  ⟨fun e =>
    let ⟨h⟩ := Quotient.exact e
    h.isEmpty,
    @mk_eq_zero α⟩


theorem mk_ne_zero_iff {α : Type u} : #α ≠ 0 ↔ Nonempty α :=
  (not_iff_not.2 mk_eq_zero_iff).trans not_isEmpty_iff


@[simp]
theorem mk_ne_zero (α : Type u) [Nonempty α] : #α ≠ 0 :=
  mk_ne_zero_iff.2 ‹_›


instance : One Cardinal.{u} :=
  -- `PUnit` might be more canonical, but this is convenient for defeq with natCast
  ⟨lift #(Fin 1)⟩


instance : Nontrivial Cardinal.{u} :=
  ⟨⟨1, 0, mk_ne_zero _⟩⟩


theorem mk_eq_one (α : Type u) [Subsingleton α] [Nonempty α] : #α = 1 :=
  let ⟨_⟩ := nonempty_unique α; (Equiv.ofUnique α (ULift (Fin 1))).cardinal_eq


theorem le_one_iff_subsingleton {α : Type u} : #α ≤ 1 ↔ Subsingleton α :=
  ⟨fun ⟨f⟩ => ⟨fun _ _ => f.injective (Subsingleton.elim _ _)⟩, fun ⟨h⟩ =>
    ⟨fun _ => ULift.up 0, fun _ _ _ => h _ _⟩⟩


@[simp]
theorem mk_le_one_iff_set_subsingleton {s : Set α} : #s ≤ 1 ↔ s.Subsingleton :=
  le_one_iff_subsingleton.trans s.subsingleton_coe


alias ⟨_, _root_.Set.Subsingleton.cardinalMk_le_one⟩ := mk_le_one_iff_set_subsingleton


@[deprecated (since := "2024-11-10")]
alias _root_.Set.Subsingleton.cardinal_mk_le_one := Set.Subsingleton.cardinalMk_le_one


instance : Add Cardinal.{u} :=
  ⟨map₂ Sum fun _ _ _ _ => Equiv.sumCongr⟩


theorem add_def (α β : Type u) : #α + #β = #(α ⊕ β) :=
  rfl


instance : NatCast Cardinal.{u} :=
  ⟨fun n => lift #(Fin n)⟩


@[simp]
theorem mk_sum (α : Type u) (β : Type v) : #(α ⊕ β) = lift.{v, u} #α + lift.{u, v} #β :=
  mk_congr (Equiv.ulift.symm.sumCongr Equiv.ulift.symm)


@[simp]
theorem mk_option {α : Type u} : #(Option α) = #α + 1 := by
  /-
    α : Type u
    ⊢ Eq (Cardinal.mk (Option α)) (HAdd.hAdd (Cardinal.mk α) 1)
  -/
  rw [(Equiv.optionEquivSumPUnit.{u, u} α).cardinal_eq, mk_sum, mk_eq_one PUnit, lift_id, lift_id]
  /-
    🎉 no goals
  -/


@[simp]
theorem mk_psum (α : Type u) (β : Type v) : #(α ⊕' β) = lift.{v} #α + lift.{u} #β :=
  (mk_congr (Equiv.psumEquivSum α β)).trans (mk_sum α β)


@[simp]
theorem mk_fintype (α : Type u) [h : Fintype α] : #α = Fintype.card α :=
                                      /-
                                        α : Type u
                                        h : Fintype α
                                        ⊢ Eq (Fintype.card α) (Fintype.card (ULift.{u, 0} (Fin (Fintype.card α))))
                                      -/
  mk_congr (Fintype.equivOfCardEq (by simp))
                                      /-
                                        🎉 no goals
                                      -/


instance : Mul Cardinal.{u} :=
  ⟨map₂ Prod fun _ _ _ _ => Equiv.prodCongr⟩


theorem mul_def (α β : Type u) : #α * #β = #(α × β) :=
  rfl


@[simp]
theorem mk_prod (α : Type u) (β : Type v) : #(α × β) = lift.{v, u} #α * lift.{u, v} #β :=
  mk_congr (Equiv.ulift.symm.prodCongr Equiv.ulift.symm)


/-- The cardinal exponential. `#α ^ #β` is the cardinal of `β → α`. -/
instance instPowCardinal : Pow Cardinal.{u} Cardinal.{u} :=
  ⟨map₂ (fun α β => β → α) fun _ _ _ _ e₁ e₂ => e₂.arrowCongr e₁⟩


theorem power_def (α β : Type u) : #α ^ #β = #(β → α) :=
  rfl


theorem mk_arrow (α : Type u) (β : Type v) : #(α → β) = (lift.{u} #β^lift.{v} #α) :=
  mk_congr (Equiv.ulift.symm.arrowCongr Equiv.ulift.symm)


@[simp]
theorem lift_power (a b : Cardinal.{u}) : lift.{v} (a ^ b) = lift.{v} a ^ lift.{v} b :=
  inductionOn₂ a b fun _ _ =>
    mk_congr <| Equiv.ulift.trans (Equiv.ulift.arrowCongr Equiv.ulift).symm


@[simp]
theorem power_zero (a : Cardinal) : a ^ (0 : Cardinal) = 1 :=
  inductionOn a fun _ => mk_eq_one _


@[simp]
theorem power_one (a : Cardinal.{u}) : a ^ (1 : Cardinal) = a :=
  inductionOn a fun α => mk_congr (Equiv.funUnique (ULift.{u} (Fin 1)) α)


theorem power_add (a b c : Cardinal) : a ^ (b + c) = a ^ b * a ^ c :=
  inductionOn₃ a b c fun α β γ => mk_congr <| Equiv.sumArrowEquivProdArrow β γ α


private theorem cast_succ (n : ℕ) : ((n + 1 : ℕ) : Cardinal.{u}) = n + 1 := by
  /-
    n : Nat
    ⊢ Eq (↑(HAdd.hAdd n 1)) (HAdd.hAdd (↑n) 1)
  -/
  change #(ULift.{u} _) = #(ULift.{u} _) + 1
  /-
    n : Nat
    ⊢ Eq (Cardinal.mk (ULift.{u, 0} (Fin (HAdd.hAdd n 1)))) (HAdd.hAdd (Cardinal.m …
  -/
  rw [← mk_option]
  /-
    n : Nat
    ⊢ Eq (Cardinal.mk (ULift.{u, 0} (Fin (HAdd.hAdd n 1)))) (Cardinal.mk (Option ( …
  -/
  simp
  /-
    🎉 no goals
  -/


instance commSemiring : CommSemiring Cardinal.{u} where
  zero := 0
  one := 1
  add := (· + ·)
  mul := (· * ·)
  zero_add a := inductionOn a fun α => mk_congr <| Equiv.emptySum _ α
  add_zero a := inductionOn a fun α => mk_congr <| Equiv.sumEmpty α _
  add_assoc a b c := inductionOn₃ a b c fun α β γ => mk_congr <| Equiv.sumAssoc α β γ
  add_comm a b := inductionOn₂ a b fun α β => mk_congr <| Equiv.sumComm α β
  zero_mul a := inductionOn a fun _ => mk_eq_zero _
  mul_zero a := inductionOn a fun _ => mk_eq_zero _
  one_mul a := inductionOn a fun α => mk_congr <| Equiv.uniqueProd α _
  mul_one a := inductionOn a fun α => mk_congr <| Equiv.prodUnique α _
  mul_assoc a b c := inductionOn₃ a b c fun α β γ => mk_congr <| Equiv.prodAssoc α β γ
  mul_comm a b := inductionOn₂ a b fun α β => mk_congr <| Equiv.prodComm α β
  left_distrib a b c := inductionOn₃ a b c fun α β γ => mk_congr <| Equiv.prodSumDistrib α β γ
  right_distrib a b c := inductionOn₃ a b c fun α β γ => mk_congr <| Equiv.sumProdDistrib α β γ
  nsmul := nsmulRec
  npow n c := c ^ (n : Cardinal)
  npow_zero := power_zero
                      /-
                        α β : Type u
                        n : Nat
                        c : Cardinal.{u}
                        ⊢ Eq ((fun n c => HPow.hPow c ↑n) (HAdd.hAdd n 1) c) (HMul.hMul ((fun n c => H …
                      -/
  npow_succ n c := by dsimp; rw [cast_succ, power_add, power_one]
                             /-
                               🎉 no goals
                             -/
  natCast n := lift #(Fin n)
  natCast_zero := rfl
  natCast_succ n := cast_succ n


@[simp]
theorem one_power {a : Cardinal} : (1 : Cardinal) ^ a = 1 :=
  inductionOn a fun _ => mk_eq_one _


                                  /-
                                    ⊢ Eq (Cardinal.mk Bool) 2
                                  -/
theorem mk_bool : #Bool = 2 := by simp
                                  /-
                                    🎉 no goals
                                  -/


                                  /-
                                    ⊢ Eq (Cardinal.mk Prop) 2
                                  -/
theorem mk_Prop : #Prop = 2 := by simp
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem zero_power {a : Cardinal} : a ≠ 0 → (0 : Cardinal) ^ a = 0 :=
  inductionOn a fun _ heq =>
    mk_eq_zero_iff.2 <|
      isEmpty_pi.2 <|
        let ⟨a⟩ := mk_ne_zero_iff.1 heq
        ⟨a, inferInstance⟩


theorem power_ne_zero {a : Cardinal} (b : Cardinal) : a ≠ 0 → a ^ b ≠ 0 :=
  inductionOn₂ a b fun _ _ h =>
    let ⟨a⟩ := mk_ne_zero_iff.1 h
    mk_ne_zero_iff.2 ⟨fun _ => a⟩


theorem mul_power {a b c : Cardinal} : (a * b) ^ c = a ^ c * b ^ c :=
  inductionOn₃ a b c fun α β γ => mk_congr <| Equiv.arrowProdEquivProdArrow α β γ


theorem power_mul {a b c : Cardinal} : a ^ (b * c) = (a ^ b) ^ c := by
  /-
    a b c : Cardinal.{u_1}
    ⊢ Eq (HPow.hPow a (HMul.hMul b c)) (HPow.hPow (HPow.hPow a b) c)
  -/
  rw [mul_comm b c]
  /-
    a b c : Cardinal.{u_1}
    ⊢ Eq (HPow.hPow a (HMul.hMul c b)) (HPow.hPow (HPow.hPow a b) c)
  -/
  exact inductionOn₃ a b c fun α β γ => mk_congr <| Equiv.curry γ β α
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem power_natCast (a : Cardinal.{u}) (n : ℕ) : a ^ (↑n : Cardinal.{u}) = a ^ n :=
  rfl


@[deprecated (since := "2024-10-16")]
alias power_cast_right := power_natCast


@[simp]
theorem lift_one : lift 1 = 1 := mk_eq_one _


@[simp]
theorem lift_eq_one {a : Cardinal.{v}} : lift.{u} a = 1 ↔ a = 1 :=
  lift_injective.eq_iff' lift_one


@[simp]
theorem lift_add (a b : Cardinal.{u}) : lift.{v} (a + b) = lift.{v} a + lift.{v} b :=
  inductionOn₂ a b fun _ _ =>
    mk_congr <| Equiv.ulift.trans (Equiv.sumCongr Equiv.ulift Equiv.ulift).symm


@[simp]
theorem lift_mul (a b : Cardinal.{u}) : lift.{v} (a * b) = lift.{v} a * lift.{v} b :=
  inductionOn₂ a b fun _ _ =>
    mk_congr <| Equiv.ulift.trans (Equiv.prodCongr Equiv.ulift Equiv.ulift).symm

-- Porting note: Proof used to be simp, needed to remind simp that 1 + 1 = 2

                                           /-
                                             ⊢ Eq (Cardinal.lift.{u, v} 2) 2
                                           -/
theorem lift_two : lift.{u, v} 2 = 2 := by simp [← one_add_one_eq_two]
                                           /-
                                             🎉 no goals
                                           -/


@[simp]
                                                      /-
                                                        α : Type u
                                                        ⊢ Eq (Cardinal.mk (Set α)) (HPow.hPow 2 (Cardinal.mk α))
                                                      -/
theorem mk_set {α : Type u} : #(Set α) = 2 ^ #α := by simp [← one_add_one_eq_two, Set, mk_arrow]
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- A variant of `Cardinal.mk_set` expressed in terms of a `Set` instead of a `Type`. -/
@[simp]
theorem mk_powerset {α : Type u} (s : Set α) : #(↥(𝒫 s)) = 2 ^ #(↥s) :=
  (mk_congr (Equiv.Set.powerset s)).trans mk_set


theorem lift_two_power (a : Cardinal) : lift.{v} (2 ^ a) = 2 ^ lift.{v} a := by
  /-
    a : Cardinal.{u_1}
    ⊢ Eq (Cardinal.lift.{v, u_1} (HPow.hPow 2 a)) (HPow.hPow 2 (Cardinal.lift.{v,  …
  -/
  simp [← one_add_one_eq_two]
  /-
    🎉 no goals
  -/


protected theorem zero_le : ∀ a : Cardinal, 0 ≤ a := by
  /-
    ⊢ ∀ (a : Cardinal.{u_1}), LE.le 0 a
  -/
  rintro ⟨α⟩
  /-
    case mk
    a✝ : Cardinal.{u_1}
    α : Type u_1
    ⊢ LE.le 0 (Quot.mk (⇑Cardinal.isEquivalent) α)
  -/
  exact ⟨Embedding.ofIsEmpty⟩
  /-
    🎉 no goals
  -/


private theorem add_le_add' : ∀ {a b c d : Cardinal}, a ≤ b → c ≤ d → a + c ≤ b + d := by
  /-
    ⊢ ∀ {a b c d : Cardinal.{u_1}}, LE.le a b → LE.le c d → LE.le (HAdd.hAdd a c)  …
  -/
  rintro ⟨α⟩ ⟨β⟩ ⟨γ⟩ ⟨δ⟩ ⟨e₁⟩ ⟨e₂⟩; exact ⟨e₁.sumMap e₂⟩
                                    /-
                                      🎉 no goals
                                    -/


instance addLeftMono : AddLeftMono Cardinal :=
  ⟨fun _ _ _ => add_le_add' le_rfl⟩


instance addRightMono : AddRightMono Cardinal :=
  ⟨fun _ _ _ h => add_le_add' h le_rfl⟩


instance canonicallyOrderedCommSemiring : CanonicallyOrderedCommSemiring Cardinal.{u} :=
  { Cardinal.commSemiring,
    Cardinal.partialOrder with
    bot := 0
    bot_le := Cardinal.zero_le
    add_le_add_left := fun _ _ => add_le_add_left
    exists_add_of_le := fun {a b} =>
      inductionOn₂ a b fun α β ⟨⟨f, hf⟩⟩ =>
        have : α ⊕ ((range f)ᶜ : Set β) ≃ β := by
          classical
          exact (Equiv.sumCongr (Equiv.ofInjective f hf) (Equiv.refl _)).trans <|
            Equiv.Set.sumCompl (range f)
        ⟨#(↥(range f)ᶜ), mk_congr this.symm⟩
    le_self_add := fun a _ => (add_zero a).ge.trans <| add_le_add_left (Cardinal.zero_le _) _
    eq_zero_or_eq_zero_of_mul_eq_zero := fun {a b} =>
      inductionOn₂ a b fun α β => by
        /-
          α✝ β✝ : Type u
          a b : Cardinal.{u}
          α β : Type u
          ⊢ Eq (HMul.hMul (Cardinal.mk α) (Cardinal.mk β)) 0 → Or (Eq (Cardinal.mk α) 0) …
        -/
        simpa only [mul_def, mk_eq_zero_iff, isEmpty_prod] using id }
        /-
          🎉 no goals
        -/


instance : CanonicallyLinearOrderedAddCommMonoid Cardinal.{u} :=
  { Cardinal.canonicallyOrderedCommSemiring, Cardinal.linearOrder with }

-- Computable instance to prevent a non-computable one being found via the one above

instance : CanonicallyOrderedAddCommMonoid Cardinal.{u} :=
  { Cardinal.canonicallyOrderedCommSemiring with }


instance : LinearOrderedCommMonoidWithZero Cardinal.{u} :=
  { Cardinal.commSemiring,
    Cardinal.linearOrder with
    mul_le_mul_left := @mul_le_mul_left' _ _ _ _
    zero_le_one := zero_le _ }

-- Computable instance to prevent a non-computable one being found via the one above

instance : CommMonoidWithZero Cardinal.{u} :=
  { Cardinal.canonicallyOrderedCommSemiring with }

-- Porting note: new
-- Computable instance to prevent a non-computable one being found via the one above

instance : CommMonoid Cardinal.{u} :=
  { Cardinal.canonicallyOrderedCommSemiring with }


theorem zero_power_le (c : Cardinal.{u}) : (0 : Cardinal.{u}) ^ c ≤ 1 := by
  /-
    c : Cardinal.{u}
    ⊢ LE.le (HPow.hPow 0 c) 1
  -/
  by_cases h : c = 0
    /-
      case pos
      c : Cardinal.{u}
      h : Eq c 0
      ⊢ LE.le (HPow.hPow 0 c) 1
    -/
  · rw [h, power_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      c : Cardinal.{u}
      h : Not (Eq c 0)
      ⊢ LE.le (HPow.hPow 0 c) 1
    -/
  · rw [zero_power h]
    /-
      case neg
      c : Cardinal.{u}
      h : Not (Eq c 0)
      ⊢ LE.le 0 1
    -/
    apply zero_le
    /-
      🎉 no goals
    -/


theorem power_le_power_left : ∀ {a b c : Cardinal}, a ≠ 0 → b ≤ c → a ^ b ≤ a ^ c := by
  /-
    ⊢ ∀ {a b c : Cardinal.{u_1}}, Ne a 0 → LE.le b c → LE.le (HPow.hPow a b) (HPow …
  -/
  rintro ⟨α⟩ ⟨β⟩ ⟨γ⟩ hα ⟨e⟩
  /-
    case mk.mk.mk.intro
    a✝ : Cardinal.{u_1}
    α : Type u_1
    b✝ : Cardinal.{u_1}
    β : Type u_1
    c✝ : Cardinal.{u_1}
    γ : Type u_1
    hα : Ne (Quot.mk (⇑Cardinal.isEquivalent) α) 0
    e : Function.Embedding β γ
    ⊢ LE.le (HPow.hPow (Quot.mk (⇑Cardinal.isEquivalent) α) (Quot.mk (⇑Cardinal.is …
  -/
  let ⟨a⟩ := mk_ne_zero_iff.1 hα
  /-
    case mk.mk.mk.intro
    a✝ : Cardinal.{u_1}
    α : Type u_1
    b✝ : Cardinal.{u_1}
    β : Type u_1
    c✝ : Cardinal.{u_1}
    γ : Type u_1
    hα : Ne (Quot.mk (⇑Cardinal.isEquivalent) α) 0
    e : Function.Embedding β γ
    a : α
    ⊢ LE.le (HPow.hPow (Quot.mk (⇑Cardinal.isEquivalent) α) (Quot.mk (⇑Cardinal.is …
  -/
  exact ⟨@Function.Embedding.arrowCongrLeft _ _ _ ⟨a⟩ e⟩
  /-
    🎉 no goals
  -/


theorem self_le_power (a : Cardinal) {b : Cardinal} (hb : 1 ≤ b) : a ≤ a ^ b := by
  /-
    a b : Cardinal.{u_1}
    hb : LE.le 1 b
    ⊢ LE.le a (HPow.hPow a b)
  -/
  rcases eq_or_ne a 0 with (rfl | ha)
    /-
      case inl
      b : Cardinal.{u_1}
      hb : LE.le 1 b
      ⊢ LE.le 0 (HPow.hPow 0 b)
    -/
  · exact zero_le _
    /-
      🎉 no goals
    -/
    /-
      case inr
      a b : Cardinal.{u_1}
      hb : LE.le 1 b
      ha : Ne a 0
      ⊢ LE.le a (HPow.hPow a b)
    -/
  · convert power_le_power_left ha hb
    /-
      case h.e'_3
      a b : Cardinal.{u_1}
      hb : LE.le 1 b
      ha : Ne a 0
      ⊢ Eq a (HPow.hPow a 1)
    -/
    exact (power_one a).symm
    /-
      🎉 no goals
    -/


/-- **Cantor's theorem** -/
theorem cantor (a : Cardinal.{u}) : a < 2 ^ a := by
  /-
    a : Cardinal.{u}
    ⊢ LT.lt a (HPow.hPow 2 a)
  -/
  induction' a using Cardinal.inductionOn with α
  /-
    case h
    α : Type u
    ⊢ LT.lt (Cardinal.mk α) (HPow.hPow 2 (Cardinal.mk α))
  -/
  rw [← mk_set]
  /-
    case h
    α : Type u
    ⊢ LT.lt (Cardinal.mk α) (Cardinal.mk (Set α))
  -/
  refine ⟨⟨⟨singleton, fun a b => singleton_eq_singleton_iff.1⟩⟩, ?_⟩
  /-
    case h
    α : Type u
    ⊢ Not (Quotient.liftOn₂ (Cardinal.mk (Set α)) (Cardinal.mk α) (fun α β => None …
  -/
  rintro ⟨⟨f, hf⟩⟩
  /-
    case h.intro.mk
    α : Type u
    f : Set α → α
    hf : Function.Injective f
    ⊢ False
  -/
  exact cantor_injective f hf
  /-
    🎉 no goals
  -/


instance : NoMaxOrder Cardinal.{u} where exists_gt a := ⟨_, cantor a⟩

-- short-circuit type class inference

instance : DistribLattice Cardinal.{u} := inferInstance


theorem one_lt_iff_nontrivial {α : Type u} : 1 < #α ↔ Nontrivial α := by
  /-
    α : Type u
    ⊢ Iff (LT.lt 1 (Cardinal.mk α)) (Nontrivial α)
  -/
  rw [← not_le, le_one_iff_subsingleton, ← not_nontrivial_iff_subsingleton, Classical.not_not]
  /-
    🎉 no goals
  -/


theorem power_le_max_power_one {a b c : Cardinal} (h : b ≤ c) : a ^ b ≤ max (a ^ c) 1 := by
  /-
    a b c : Cardinal.{u_1}
    h : LE.le b c
    ⊢ LE.le (HPow.hPow a b) (Max.max (HPow.hPow a c) 1)
  -/
  by_cases ha : a = 0
    /-
      case pos
      a b c : Cardinal.{u_1}
      h : LE.le b c
      ha : Eq a 0
      ⊢ LE.le (HPow.hPow a b) (Max.max (HPow.hPow a c) 1)
    -/
  · simp [ha, zero_power_le]
    /-
      🎉 no goals
    -/
    /-
      case neg
      a b c : Cardinal.{u_1}
      h : LE.le b c
      ha : Not (Eq a 0)
      ⊢ LE.le (HPow.hPow a b) (Max.max (HPow.hPow a c) 1)
    -/
  · exact (power_le_power_left ha h).trans (le_max_left _ _)
    /-
      🎉 no goals
    -/


theorem power_le_power_right {a b c : Cardinal} : a ≤ b → a ^ c ≤ b ^ c :=
  inductionOn₃ a b c fun _ _ _ ⟨e⟩ => ⟨Embedding.arrowCongrRight e⟩


theorem power_pos {a : Cardinal} (b : Cardinal) (ha : 0 < a) : 0 < a ^ b :=
  (power_ne_zero _ ha.ne').bot_lt


protected theorem lt_wf : @WellFounded Cardinal.{u} (· < ·) :=
  ⟨fun a =>
    by_contradiction fun h => by
      /-
        a : Cardinal.{u}
        h : Not (Acc (fun x1 x2 => LT.lt x1 x2) a)
        ⊢ False
      -/
      let ι := { c : Cardinal // ¬Acc (· < ·) c }
      /-
        a : Cardinal.{u}
        h : Not (Acc (fun x1 x2 => LT.lt x1 x2) a)
        ι : Type (max 0 (?u.56267 + 1)) := Subtype fun c => Not (Acc (fun x1 x2 => LT. …
        ⊢ False
      -/
      let f : ι → Cardinal := Subtype.val
      /-
        a : Cardinal.{u}
        h : Not (Acc (fun x1 x2 => LT.lt x1 x2) a)
        ι : Type (max 0 (?u.56267 + 1)) := Subtype fun c => Not (Acc (fun x1 x2 => LT. …
        f : ι → Cardinal.{?u.56267} := Subtype.val
        ⊢ False
      -/
      haveI hι : Nonempty ι := ⟨⟨_, h⟩⟩
      obtain ⟨⟨c : Cardinal, hc : ¬Acc (· < ·) c⟩, ⟨h_1 : ∀ j, (f ⟨c, hc⟩).out ↪ (f j).out⟩⟩ :=
        Embedding.min_injective fun i => (f i).out
      /-
        case intro.mk.intro
        a : Cardinal.{u}
        h : Not (Acc (fun x1 x2 => LT.lt x1 x2) a)
        ι : Type (max 0 (u + 1)) := Subtype fun c => Not (Acc (fun x1 x2 => LT.lt x1 x …
        f : ι → Cardinal.{u} := Subtype.val
        hι : Nonempty ι
        c : Cardinal.{u}
        hc : Not (Acc (fun x1 x2 => LT.lt x1 x2) c)
        h_1 : (j : ι) → Function.Embedding (Quotient.out (f ⟨c, hc⟩)) (Quotient.out (f …
        ⊢ False
      -/
      refine hc (Acc.intro _ fun j h' => by_contradiction fun hj => h'.2 ?_)
      /-
        case intro.mk.intro
        a : Cardinal.{u}
        h : Not (Acc (fun x1 x2 => LT.lt x1 x2) a)
        ι : Type (max 0 (u + 1)) := Subtype fun c => Not (Acc (fun x1 x2 => LT.lt x1 x …
        f : ι → Cardinal.{u} := Subtype.val
        hι : Nonempty ι
        c : Cardinal.{u}
        hc : Not (Acc (fun x1 x2 => LT.lt x1 x2) c)
        h_1 : (j : ι) → Function.Embedding (Quotient.out (f ⟨c, hc⟩)) (Quotient.out (f …
        j : Cardinal.{u}
        h' : LT.lt j c
        hj : Not (Acc (fun x1 x2 => LT.lt x1 x2) j)
        ⊢ Quotient.liftOn₂ c j (fun α β => Nonempty (Function.Embedding α β)) Cardinal …
      -/
      have : #_ ≤ #_ := ⟨h_1 ⟨j, hj⟩⟩
      /-
        case intro.mk.intro
        a : Cardinal.{u}
        h : Not (Acc (fun x1 x2 => LT.lt x1 x2) a)
        ι : Type (max 0 (u + 1)) := Subtype fun c => Not (Acc (fun x1 x2 => LT.lt x1 x …
        f : ι → Cardinal.{u} := Subtype.val
        hι : Nonempty ι
        c : Cardinal.{u}
        hc : Not (Acc (fun x1 x2 => LT.lt x1 x2) c)
        h_1 : (j : ι) → Function.Embedding (Quotient.out (f ⟨c, hc⟩)) (Quotient.out (f …
        j : Cardinal.{u}
        h' : LT.lt j c
        hj : Not (Acc (fun x1 x2 => LT.lt x1 x2) j)
        this : LE.le (Cardinal.mk (Quotient.out (f ⟨c, hc⟩))) (Cardinal.mk (Quotient.o …
        ⊢ Quotient.liftOn₂ c j (fun α β => Nonempty (Function.Embedding α β)) Cardinal …
      -/
      simpa only [mk_out] using this⟩
      /-
        🎉 no goals
      -/


instance : WellFoundedRelation Cardinal.{u} :=
  ⟨(· < ·), Cardinal.lt_wf⟩

-- Porting note: this no longer is automatically inferred.

instance : WellFoundedLT Cardinal.{u} :=
  ⟨Cardinal.lt_wf⟩


instance : ConditionallyCompleteLinearOrderBot Cardinal :=
  WellFoundedLT.conditionallyCompleteLinearOrderBot _


@[simp]
theorem sInf_empty : sInf (∅ : Set Cardinal.{u}) = 0 :=
  dif_neg Set.not_nonempty_empty


lemma sInf_eq_zero_iff {s : Set Cardinal} : sInf s = 0 ↔ s = ∅ ∨ ∃ a ∈ s, a = 0 := by
  /-
    s : Set Cardinal.{u_1}
    ⊢ Iff (Eq (InfSet.sInf s) 0) (Or (Eq s EmptyCollection.emptyCollection) (Exist …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      s : Set Cardinal.{u_1}
      h : Eq (InfSet.sInf s) 0
      ⊢ Or (Eq s EmptyCollection.emptyCollection) (Exists fun a => And (Membership.m …
    -/
  · rcases s.eq_empty_or_nonempty with rfl | hne
      /-
        case refine_1.inl
        h : Eq (InfSet.sInf EmptyCollection.emptyCollection) 0
        ⊢ Or (Eq EmptyCollection.emptyCollection EmptyCollection.emptyCollection) (Exi …
      -/
    · exact Or.inl rfl
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr
        s : Set Cardinal.{u_1}
        h : Eq (InfSet.sInf s) 0
        hne : s.Nonempty
        ⊢ Or (Eq s EmptyCollection.emptyCollection) (Exists fun a => And (Membership.m …
      -/
    · exact Or.inr ⟨sInf s, csInf_mem hne, h⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      s : Set Cardinal.{u_1}
      h : Or (Eq s EmptyCollection.emptyCollection) (Exists fun a => And (Membership …
      ⊢ Eq (InfSet.sInf s) 0
    -/
  · rcases h with rfl | ⟨a, ha, rfl⟩
      /-
        case refine_2.inl
        ⊢ Eq (InfSet.sInf EmptyCollection.emptyCollection) 0
      -/
    · exact Cardinal.sInf_empty
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr.intro.intro
        s : Set Cardinal.{u_1}
        ha : Membership.mem s 0
        ⊢ Eq (InfSet.sInf s) 0
      -/
    · exact eq_bot_iff.2 (csInf_le' ha)
      /-
        🎉 no goals
      -/


lemma iInf_eq_zero_iff {ι : Sort*} {f : ι → Cardinal} :
    (⨅ i, f i) = 0 ↔ IsEmpty ι ∨ ∃ i, f i = 0 := by
  /-
    ι : Sort u_1
    f : ι → Cardinal.{u_2}
    ⊢ Iff (Eq (iInf fun i => f i) 0) (Or (IsEmpty ι) (Exists fun i => Eq (f i) 0))
  -/
  simp [iInf, sInf_eq_zero_iff]
  /-
    🎉 no goals
  -/


/-- A variant of `ciSup_of_empty` but with `0` on the RHS for convenience -/
protected theorem iSup_of_empty {ι} (f : ι → Cardinal) [IsEmpty ι] : iSup f = 0 :=
  ciSup_of_empty f


@[simp]
theorem lift_sInf (s : Set Cardinal) : lift.{u, v} (sInf s) = sInf (lift.{u, v} '' s) := by
  /-
    s : Set Cardinal.{v}
    ⊢ Eq (Cardinal.lift.{u, v} (InfSet.sInf s)) (InfSet.sInf (Set.image Cardinal.l …
  -/
  rcases eq_empty_or_nonempty s with (rfl | hs)
    /-
      case inl
      ⊢ Eq (Cardinal.lift.{u, v} (InfSet.sInf EmptyCollection.emptyCollection)) (Inf …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      s : Set Cardinal.{v}
      hs : s.Nonempty
      ⊢ Eq (Cardinal.lift.{u, v} (InfSet.sInf s)) (InfSet.sInf (Set.image Cardinal.l …
    -/
  · exact lift_monotone.map_csInf hs
    /-
      🎉 no goals
    -/


@[simp]
theorem lift_iInf {ι} (f : ι → Cardinal) : lift.{u, v} (iInf f) = ⨅ i, lift.{u, v} (f i) := by
  /-
    ι : Sort u_1
    f : ι → Cardinal.{v}
    ⊢ Eq (Cardinal.lift.{u, v} (iInf f)) (iInf fun i => Cardinal.lift.{u, v} (f i))
  -/
  unfold iInf
  /-
    ι : Sort u_1
    f : ι → Cardinal.{v}
    ⊢ Eq (Cardinal.lift.{u, v} (InfSet.sInf (Set.range f))) (InfSet.sInf (Set.rang …
  -/
  convert lift_sInf (range f)
  /-
    case h.e'_3.h.e'_3
    ι : Sort u_1
    f : ι → Cardinal.{v}
    ⊢ Eq (Set.range fun i => Cardinal.lift.{u, v} (f i)) (Set.image Cardinal.lift. …
  -/
  simp_rw [← comp_apply (f := lift), range_comp]
  /-
    🎉 no goals
  -/


/-- Note that the successor of `c` is not the same as `c + 1` except in the case of finite `c`. -/
instance : SuccOrder Cardinal := ConditionallyCompleteLinearOrder.toSuccOrder


theorem succ_def (c : Cardinal) : succ c = sInf { c' | c < c' } :=
  dif_neg <| not_isMax c


theorem succ_pos : ∀ c : Cardinal, 0 < succ c :=
  bot_lt_succ


theorem succ_ne_zero (c : Cardinal) : succ c ≠ 0 :=
  (succ_pos _).ne'


theorem add_one_le_succ (c : Cardinal.{u}) : c + 1 ≤ succ c := by
  -- Porting note: rewrote the next three lines to avoid defeq abuse.
  /-
    c : Cardinal.{u}
    ⊢ LE.le (HAdd.hAdd c 1) (Order.succ c)
  -/
  have : Set.Nonempty { c' | c < c' } := exists_gt c
  /-
    c : Cardinal.{u}
    this : (setOf fun c' => LT.lt c c').Nonempty
    ⊢ LE.le (HAdd.hAdd c 1) (Order.succ c)
  -/
  simp_rw [succ_def, le_csInf_iff'' this, mem_setOf]
  /-
    c : Cardinal.{u}
    this : (setOf fun c' => LT.lt c c').Nonempty
    ⊢ ∀ (b : Cardinal.{u}), LT.lt c b → LE.le (HAdd.hAdd c 1) b
  -/
  intro b hlt
  /-
    c : Cardinal.{u}
    this : (setOf fun c' => LT.lt c c').Nonempty
    b : Cardinal.{u}
    hlt : LT.lt c b
    ⊢ LE.le (HAdd.hAdd c 1) b
  -/
  rcases b, c with ⟨⟨β⟩, ⟨γ⟩⟩
  /-
    case mk.mk
    c b : Cardinal.{u}
    β γ : Type u
    this : (setOf fun c' => LT.lt (Quot.mk (⇑Cardinal.isEquivalent) γ) c').Nonempty
    hlt : LT.lt (Quot.mk (⇑Cardinal.isEquivalent) γ) (Quot.mk (⇑Cardinal.isEquival …
    ⊢ LE.le (HAdd.hAdd (Quot.mk (⇑Cardinal.isEquivalent) γ) 1) (Quot.mk (⇑Cardinal …
  -/
  cases' le_of_lt hlt with f
  /-
    case mk.mk.intro
    c b : Cardinal.{u}
    β γ : Type u
    this : (setOf fun c' => LT.lt (Quot.mk (⇑Cardinal.isEquivalent) γ) c').Nonempty
    hlt : LT.lt (Quot.mk (⇑Cardinal.isEquivalent) γ) (Quot.mk (⇑Cardinal.isEquival …
    f : Function.Embedding γ β
    ⊢ LE.le (HAdd.hAdd (Quot.mk (⇑Cardinal.isEquivalent) γ) 1) (Quot.mk (⇑Cardinal …
  -/
  have : ¬Surjective f := fun hn => (not_le_of_lt hlt) (mk_le_of_surjective hn)
  /-
    case mk.mk.intro
    c b : Cardinal.{u}
    β γ : Type u
    this✝ : (setOf fun c' => LT.lt (Quot.mk (⇑Cardinal.isEquivalent) γ) c').Nonempty
    hlt : LT.lt (Quot.mk (⇑Cardinal.isEquivalent) γ) (Quot.mk (⇑Cardinal.isEquival …
    f : Function.Embedding γ β
    this : Not (Function.Surjective ⇑f)
    ⊢ LE.le (HAdd.hAdd (Quot.mk (⇑Cardinal.isEquivalent) γ) 1) (Quot.mk (⇑Cardinal …
  -/
  simp only [Surjective, not_forall] at this
  /-
    case mk.mk.intro
    c b : Cardinal.{u}
    β γ : Type u
    this✝ : (setOf fun c' => LT.lt (Quot.mk (⇑Cardinal.isEquivalent) γ) c').Nonempty
    hlt : LT.lt (Quot.mk (⇑Cardinal.isEquivalent) γ) (Quot.mk (⇑Cardinal.isEquival …
    f : Function.Embedding γ β
    this : Exists fun x => Not (Exists fun a => Eq (f a) x)
    ⊢ LE.le (HAdd.hAdd (Quot.mk (⇑Cardinal.isEquivalent) γ) 1) (Quot.mk (⇑Cardinal …
  -/
  rcases this with ⟨b, hb⟩
  calc
    #γ + 1 = #(Option γ) := mk_option.symm
    _ ≤ #β := (f.optionElim b hb).cardinal_le


@[simp]
theorem lift_succ (a) : lift.{v, u} (succ a) = succ (lift.{v, u} a) :=
  le_antisymm
    (le_of_not_gt fun h => by
      /-
        a : Cardinal.{u}
        h : GT.gt (Cardinal.lift.{v, u} (Order.succ a)) (Order.succ (Cardinal.lift.{v, …
        ⊢ False
      -/
      rcases lt_lift_iff.1 h with ⟨b, h, e⟩
      /-
        case intro.intro
        a : Cardinal.{u}
        h✝ : GT.gt (Cardinal.lift.{v, u} (Order.succ a)) (Order.succ (Cardinal.lift.{v …
        b : Cardinal.{u}
        h : LT.lt b (Order.succ a)
        e : Eq (Cardinal.lift.{v, u} b) (Order.succ (Cardinal.lift.{v, u} a))
        ⊢ False
      -/
      rw [lt_succ_iff, ← lift_le, e] at h
      /-
        case intro.intro
        a : Cardinal.{u}
        h✝ : GT.gt (Cardinal.lift.{v, u} (Order.succ a)) (Order.succ (Cardinal.lift.{v …
        b : Cardinal.{u}
        h : LE.le (Order.succ (Cardinal.lift.{v, u} a)) (Cardinal.lift.{v, u} a)
        e : Eq (Cardinal.lift.{v, u} b) (Order.succ (Cardinal.lift.{v, u} a))
        ⊢ False
      -/
      exact h.not_lt (lt_succ _))
      /-
        🎉 no goals
      -/
    (succ_le_of_lt <| lift_lt.2 <| lt_succ a)


/-- A cardinal is a limit if it is not zero or a successor cardinal. Note that `ℵ₀` is a limit
  cardinal by this definition, but `0` isn't.
Deprecated. Use `Order.IsSuccLimit` instead. -/
@[deprecated IsSuccLimit (since := "2024-09-17")]
def IsLimit (c : Cardinal) : Prop :=
  c ≠ 0 ∧ IsSuccPrelimit c


theorem ne_zero_of_isSuccLimit {c} (h : IsSuccLimit c) : c ≠ 0 :=
  h.ne_bot


theorem isSuccPrelimit_zero : IsSuccPrelimit (0 : Cardinal) :=
  isSuccPrelimit_bot


protected theorem isSuccLimit_iff {c : Cardinal} : IsSuccLimit c ↔ c ≠ 0 ∧ IsSuccPrelimit c :=
  isSuccLimit_iff


set_option linter.deprecated false in
@[deprecated IsSuccLimit.isSuccPrelimit (since := "2024-09-17")]
protected theorem IsLimit.isSuccPrelimit {c} (h : IsLimit c) : IsSuccPrelimit c :=
  h.2


set_option linter.deprecated false in
@[deprecated ne_zero_of_isSuccLimit (since := "2024-09-17")]
protected theorem IsLimit.ne_zero {c} (h : IsLimit c) : c ≠ 0 :=
  h.1


set_option linter.deprecated false in
@[deprecated IsLimit.isSuccPrelimit (since := "2024-09-05")]
alias IsLimit.isSuccLimit := IsLimit.isSuccPrelimit


set_option linter.deprecated false in
@[deprecated IsSuccLimit.succ_lt (since := "2024-09-17")]
theorem IsLimit.succ_lt {x c} (h : IsLimit c) : x < c → succ x < c :=
  h.isSuccPrelimit.succ_lt


@[deprecated isSuccPrelimit_zero (since := "2024-09-05")]
alias isSuccLimit_zero := isSuccPrelimit_zero


/-- The indexed sum of cardinals is the cardinality of the
  indexed disjoint union, i.e. sigma type. -/
def sum {ι} (f : ι → Cardinal) : Cardinal :=
  mk (Σ i, (f i).out)


theorem le_sum {ι} (f : ι → Cardinal) (i) : f i ≤ sum f := by
  /-
    ι : Type u_1
    f : ι → Cardinal.{max u_1 u_2}
    i : ι
    ⊢ LE.le (f i) (Cardinal.sum f)
  -/
  rw [← Quotient.out_eq (f i)]
  /-
    ι : Type u_1
    f : ι → Cardinal.{max u_1 u_2}
    i : ι
    ⊢ LE.le (Quotient.mk Cardinal.isEquivalent (Quotient.out (f i))) (Cardinal.sum …
  -/
  exact ⟨⟨fun a => ⟨i, a⟩, fun a b h => by injection h⟩⟩
  /-
    🎉 no goals
  -/


theorem iSup_le_sum {ι} (f : ι → Cardinal) : iSup f ≤ sum f :=
  ciSup_le' <| le_sum _


@[simp]
theorem mk_sigma {ι} (f : ι → Type*) : #(Σ i, f i) = sum fun i => #(f i) :=
  mk_congr <| Equiv.sigmaCongrRight fun _ => outMkEquiv.symm


theorem mk_sigma_congr_lift {ι : Type v} {ι' : Type v'} {f : ι → Type w} {g : ι' → Type w'}
    (e : ι ≃ ι') (h : ∀ i, lift.{w'} #(f i) = lift.{w} #(g (e i))) :
    lift.{max v' w'} #(Σ i, f i) = lift.{max v w} #(Σ i, g i) :=
  Cardinal.lift_mk_eq'.2 ⟨.sigmaCongr e fun i ↦ Classical.choice <| Cardinal.lift_mk_eq'.1 (h i)⟩


theorem mk_sigma_congr {ι ι' : Type u} {f : ι → Type v} {g : ι' → Type v} (e : ι ≃ ι')
    (h : ∀ i, #(f i) = #(g (e i))) : #(Σ i, f i) = #(Σ i, g i) :=
  mk_congr <| Equiv.sigmaCongr e fun i ↦ Classical.choice <| Cardinal.eq.mp (h i)


/-- Similar to `mk_sigma_congr` with indexing types in different universes. This is not a strict
generalization. -/
theorem mk_sigma_congr' {ι : Type u} {ι' : Type v} {f : ι → Type max w (max u v)}
    {g : ι' → Type max w (max u v)} (e : ι ≃ ι')
    (h : ∀ i, #(f i) = #(g (e i))) : #(Σ i, f i) = #(Σ i, g i) :=
  mk_congr <| Equiv.sigmaCongr e fun i ↦ Classical.choice <| Cardinal.eq.mp (h i)


theorem mk_sigma_congrRight {ι : Type u} {f g : ι → Type v} (h : ∀ i, #(f i) = #(g i)) :
    #(Σ i, f i) = #(Σ i, g i) :=
  mk_sigma_congr (Equiv.refl ι) h


theorem mk_psigma_congrRight {ι : Type u} {f g : ι → Type v} (h : ∀ i, #(f i) = #(g i)) :
    #(Σ' i, f i) = #(Σ' i, g i) :=
  mk_congr <| .psigmaCongrRight fun i => Classical.choice <| Cardinal.eq.mp (h i)


theorem mk_psigma_congrRight_prop {ι : Prop} {f g : ι → Type v} (h : ∀ i, #(f i) = #(g i)) :
    #(Σ' i, f i) = #(Σ' i, g i) :=
  mk_congr <| .psigmaCongrRight fun i => Classical.choice <| Cardinal.eq.mp (h i)


theorem mk_sigma_arrow {ι} (α : Type*) (f : ι → Type*) :
    #(Sigma f → α) = #(Π i, f i → α) := mk_congr <| Equiv.piCurry fun _ _ ↦ α


@[simp]
theorem sum_const (ι : Type u) (a : Cardinal.{v}) :
    (sum fun _ : ι => a) = lift.{v} #ι * lift.{u} a :=
  inductionOn a fun α =>
    mk_congr <|
      calc
        (Σ _ : ι, Quotient.out #α) ≃ ι × Quotient.out #α := Equiv.sigmaEquivProd _ _
        _ ≃ ULift ι × ULift α := Equiv.ulift.symm.prodCongr (outMkEquiv.trans Equiv.ulift.symm)


                                                                                         /-
                                                                                           ι : Type u
                                                                                           a : Cardinal.{u}
                                                                                           ⊢ Eq (Cardinal.sum fun x => a) (HMul.hMul (Cardinal.mk ι) a)
                                                                                         -/
theorem sum_const' (ι : Type u) (a : Cardinal.{u}) : (sum fun _ : ι => a) = #ι * a := by simp
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


@[simp]
theorem sum_add_distrib {ι} (f g : ι → Cardinal) : sum (f + g) = sum f + sum g := by
  /-
    ι : Type u_1
    f g : ι → Cardinal.{u_2}
    ⊢ Eq (Cardinal.sum (HAdd.hAdd f g)) (HAdd.hAdd (Cardinal.sum f) (Cardinal.sum  …
  -/
  have := mk_congr (Equiv.sigmaSumDistrib (Quotient.out ∘ f) (Quotient.out ∘ g))
  /-
    ι : Type u_1
    f g : ι → Cardinal.{u_2}
    this : Eq (Cardinal.mk (Sigma fun i => Sum (Function.comp Quotient.out f i) (F …
    ⊢ Eq (Cardinal.sum (HAdd.hAdd f g)) (HAdd.hAdd (Cardinal.sum f) (Cardinal.sum  …
  -/
  simp only [comp_apply, mk_sigma, mk_sum, mk_out, lift_id] at this
  /-
    ι : Type u_1
    f g : ι → Cardinal.{u_2}
    this : Eq (Cardinal.sum fun i => HAdd.hAdd (f i) (g i)) (HAdd.hAdd (Cardinal.s …
    ⊢ Eq (Cardinal.sum (HAdd.hAdd f g)) (HAdd.hAdd (Cardinal.sum f) (Cardinal.sum  …
  -/
  exact this
  /-
    🎉 no goals
  -/


@[simp]
theorem sum_add_distrib' {ι} (f g : ι → Cardinal) :
    (Cardinal.sum fun i => f i + g i) = sum f + sum g :=
  sum_add_distrib f g


@[simp]
theorem lift_sum {ι : Type u} (f : ι → Cardinal.{v}) :
    Cardinal.lift.{w} (Cardinal.sum f) = Cardinal.sum fun i => Cardinal.lift.{w} (f i) :=
  Equiv.cardinal_eq <|
    Equiv.ulift.trans <|
      Equiv.sigmaCongrRight fun a =>
    -- Porting note: Inserted universe hint .{_,_,v} below
                            /-
                              ι : Type u
                              f : ι → Cardinal.{v}
                              a : ι
                              ⊢ Nonempty (Equiv (Quotient.out (f a)) (Quotient.out ((fun i => Cardinal.lift. …
                            -/
        Nonempty.some <| by rw [← lift_mk_eq.{_,_,v}, mk_out, mk_out, lift_lift]
                            /-
                              🎉 no goals
                            -/


theorem sum_le_sum {ι} (f g : ι → Cardinal) (H : ∀ i, f i ≤ g i) : sum f ≤ sum g :=
  ⟨(Embedding.refl _).sigmaMap fun i =>
                             /-
                               ι : Type u_1
                               f g : ι → Cardinal.{u_2}
                               H : ∀ (i : ι), LE.le (f i) (g i)
                               i : ι
                               ⊢ Nonempty (Function.Embedding (Quotient.out (f i)) (Quotient.out (g ((Functio …
                             -/
      Classical.choice <| by have := H i; rwa [← Quot.out_eq (f i), ← Quot.out_eq (g i)] at this⟩
                                          /-
                                            🎉 no goals
                                          -/


theorem mk_le_mk_mul_of_mk_preimage_le {c : Cardinal} (f : α → β) (hf : ∀ b : β, #(f ⁻¹' {b}) ≤ c) :
    #α ≤ #β * c := by
  simpa only [← mk_congr (@Equiv.sigmaFiberEquiv α β f), mk_sigma, ← sum_const'] using
    sum_le_sum _ _ hf


theorem lift_mk_le_lift_mk_mul_of_lift_mk_preimage_le {α : Type u} {β : Type v} {c : Cardinal}
    (f : α → β) (hf : ∀ b : β, lift.{v} #(f ⁻¹' {b}) ≤ c) : lift.{v} #α ≤ lift.{u} #β * c :=
  (mk_le_mk_mul_of_mk_preimage_le fun x : ULift.{v} α => ULift.up.{u} (f x.1)) <|
    ULift.forall.2 fun b =>
      (mk_congr <|
            (Equiv.ulift.image _).trans
              (Equiv.trans
                (by
                  /-
                    α : Type u
                    β : Type v
                    c : Cardinal.{max u v}
                    f : α → β
                    hf : ∀ (b : β), LE.le (Cardinal.lift.{v, u} (Cardinal.mk ↑(Set.preimage f (Sin …
                    b : β
                    ⊢ Equiv ↑(Set.image (⇑Equiv.ulift) (Set.preimage (fun x => { down := f x.down  …
                  -/
                  rw [Equiv.image_eq_preimage]
                  /- Porting note: Need to insert the following `have` b/c bad fun coercion
                   behaviour for Equivs -/
                  /-
                    α : Type u
                    β : Type v
                    c : Cardinal.{max u v}
                    f : α → β
                    hf : ∀ (b : β), LE.le (Cardinal.lift.{v, u} (Cardinal.mk ↑(Set.preimage f (Sin …
                    b : β
                    ⊢ Equiv ↑(Set.preimage (⇑Equiv.ulift.symm) (Set.preimage (fun x => { down := f …
                  -/
                  have : DFunLike.coe (Equiv.symm (Equiv.ulift (α := α))) = ULift.up (α := α) := rfl
                  /-
                    α : Type u
                    β : Type v
                    c : Cardinal.{max u v}
                    f : α → β
                    hf : ∀ (b : β), LE.le (Cardinal.lift.{v, u} (Cardinal.mk ↑(Set.preimage f (Sin …
                    b : β
                    this : Eq (⇑Equiv.ulift.symm) ULift.up
                    ⊢ Equiv ↑(Set.preimage (⇑Equiv.ulift.symm) (Set.preimage (fun x => { down := f …
                  -/
                  rw [this]
                  /-
                    α : Type u
                    β : Type v
                    c : Cardinal.{max u v}
                    f : α → β
                    hf : ∀ (b : β), LE.le (Cardinal.lift.{v, u} (Cardinal.mk ↑(Set.preimage f (Sin …
                    b : β
                    this : Eq (⇑Equiv.ulift.symm) ULift.up
                    ⊢ Equiv ↑(Set.preimage ULift.up (Set.preimage (fun x => { down := f x.down })  …
                  -/
                  simp only [preimage, mem_singleton_iff, ULift.up_inj, mem_setOf_eq, coe_setOf]
                  /-
                    α : Type u
                    β : Type v
                    c : Cardinal.{max u v}
                    f : α → β
                    hf : ∀ (b : β), LE.le (Cardinal.lift.{v, u} (Cardinal.mk ↑(Set.preimage f (Sin …
                    b : β
                    this : Eq (⇑Equiv.ulift.symm) ULift.up
                    ⊢ Equiv (Subtype fun x => Eq (f x) b) (Subtype fun x => Eq (f x) b)
                  -/
                  exact Equiv.refl _)
                  /-
                    🎉 no goals
                  -/
                Equiv.ulift.symm)).trans_le
        (hf b)


theorem sum_nat_eq_add_sum_succ (f : ℕ → Cardinal.{u}) :
    Cardinal.sum f = f 0 + Cardinal.sum fun i => f (i + 1) := by
  /-
    f : Nat → Cardinal.{u}
    ⊢ Eq (Cardinal.sum f) (HAdd.hAdd (f 0) (Cardinal.sum fun i => f (HAdd.hAdd i 1 …
  -/
  refine (Equiv.sigmaNatSucc fun i => Quotient.out (f i)).cardinal_eq.trans ?_
  /-
    f : Nat → Cardinal.{u}
    ⊢ Eq (Cardinal.mk (Sum (Quotient.out (f 0)) (Sigma fun n => Quotient.out (f (H …
  -/
  simp only [mk_sum, mk_out, lift_id, mk_sigma]
  /-
    🎉 no goals
  -/


open Cardinal in
theorem nonempty_embedding_to_cardinal : Nonempty (α ↪ Cardinal.{u}) :=
  (Embedding.total _ _).resolve_left fun ⟨⟨f, hf⟩⟩ =>
    let g : α → Cardinal.{u} := invFun f
    let ⟨x, (hx : g x = 2 ^ sum g)⟩ := invFun_surjective hf (2 ^ sum g)
    have : g x ≤ sum g := le_sum.{u, u} g x
                     /-
                       α : Type u
                       x✝ : Nonempty (Function.Embedding Cardinal.{u} α)
                       f : Cardinal.{u} → α
                       hf : Function.Injective f
                       g : α → Cardinal.{u} := Function.invFun f
                       x : α
                       hx : Eq (g x) (HPow.hPow 2 (Cardinal.sum g))
                       this : LE.le (g x) (Cardinal.sum g)
                       ⊢ GT.gt (g x) (Cardinal.sum g)
                     -/
    not_le_of_gt (by rw [hx]; exact cantor _) this
                              /-
                                🎉 no goals
                              -/


/-- An embedding of any type to the set of cardinals in its universe. -/
def embeddingToCardinal : α ↪ Cardinal.{u} :=
  Classical.choice nonempty_embedding_to_cardinal


/-- Any type can be endowed with a well order, obtained by pulling back the well order over
cardinals by some embedding. -/
def WellOrderingRel : α → α → Prop :=
  embeddingToCardinal ⁻¹'o (· < ·)


instance WellOrderingRel.isWellOrder : IsWellOrder α WellOrderingRel :=
  (RelEmbedding.preimage _ _).isWellOrder


instance IsWellOrder.subtype_nonempty : Nonempty { r // IsWellOrder α r } :=
  ⟨⟨WellOrderingRel, inferInstance⟩⟩


variable (α) in
/-- The **well-ordering theorem** (or **Zermelo's theorem**): every type has a well-order -/
theorem exists_wellOrder : ∃ (_ : LinearOrder α), WellFoundedLT α := by
  classical
  exact ⟨linearOrderOfSTO WellOrderingRel, WellOrderingRel.isWellOrder.toIsWellFounded⟩


instance small_Iic (a : Cardinal.{u}) : Small.{u} (Iic a) := by
  /-
    α β : Type u
    a : Cardinal.{u}
    ⊢ Small.{u, u + 1} ↑(Set.Iic a)
  -/
  rw [← mk_out a]
  /-
    α β : Type u
    a : Cardinal.{u}
    ⊢ Small.{u, u + 1} ↑(Set.Iic (Cardinal.mk (Quotient.out a)))
  -/
  apply @small_of_surjective (Set a.out) (Iic #a.out) _ fun x => ⟨#x, mk_set_le x⟩
  /-
    α β : Type u
    a : Cardinal.{u}
    ⊢ Function.Surjective fun x => ⟨Cardinal.mk ↑x, ⋯⟩
  -/
  rintro ⟨x, hx⟩
  /-
    case mk
    α β : Type u
    a x : Cardinal.{u}
    hx : Membership.mem (Set.Iic (Cardinal.mk (Quotient.out a))) x
    ⊢ Exists fun a_1 => Eq ((fun x => ⟨Cardinal.mk ↑x, ⋯⟩) a_1) ⟨x, hx⟩
  -/
  simpa using le_mk_iff_exists_set.1 hx
  /-
    🎉 no goals
  -/


instance small_Iio (a : Cardinal.{u}) : Small.{u} (Iio a) := small_subset Iio_subset_Iic_self

instance small_Icc (a b : Cardinal.{u}) : Small.{u} (Icc a b) := small_subset Icc_subset_Iic_self

instance small_Ico (a b : Cardinal.{u}) : Small.{u} (Ico a b) := small_subset Ico_subset_Iio_self

instance small_Ioc (a b : Cardinal.{u}) : Small.{u} (Ioc a b) := small_subset Ioc_subset_Iic_self

instance small_Ioo (a b : Cardinal.{u}) : Small.{u} (Ioo a b) := small_subset Ioo_subset_Iio_self


/-- A set of cardinals is bounded above iff it's small, i.e. it corresponds to a usual ZFC set. -/
theorem bddAbove_iff_small {s : Set Cardinal.{u}} : BddAbove s ↔ Small.{u} s :=
  ⟨fun ⟨a, ha⟩ => @small_subset _ (Iic a) s (fun _ h => ha h) _, by
    /-
      s : Set Cardinal.{u}
      ⊢ Small.{u, u + 1} ↑s → BddAbove s
    -/
    rintro ⟨ι, ⟨e⟩⟩
    /-
      case mk.intro.intro
      s : Set Cardinal.{u}
      ι : Type u
      e : Equiv (↑s) ι
      ⊢ BddAbove s
    -/
    use sum.{u, u} fun x ↦ e.symm x
    /-
      case h
      s : Set Cardinal.{u}
      ι : Type u
      e : Equiv (↑s) ι
      ⊢ Membership.mem (upperBounds s) (Cardinal.sum fun x => ↑(e.symm x))
    -/
    intro a ha
    /-
      case h
      s : Set Cardinal.{u}
      ι : Type u
      e : Equiv (↑s) ι
      a : Cardinal.{u}
      ha : Membership.mem s a
      ⊢ LE.le a (Cardinal.sum fun x => ↑(e.symm x))
    -/
    simpa using le_sum (fun x ↦ e.symm x) (e ⟨a, ha⟩)⟩
    /-
      🎉 no goals
    -/


theorem bddAbove_of_small (s : Set Cardinal.{u}) [h : Small.{u} s] : BddAbove s :=
  bddAbove_iff_small.2 h


theorem bddAbove_range {ι : Type*} [Small.{u} ι] (f : ι → Cardinal.{u}) : BddAbove (Set.range f) :=
  bddAbove_of_small _


theorem bddAbove_image (f : Cardinal.{u} → Cardinal.{max u v}) {s : Set Cardinal.{u}}
    (hs : BddAbove s) : BddAbove (f '' s) := by
  /-
    f : Cardinal.{u} → Cardinal.{max u v}
    s : Set Cardinal.{u}
    hs : BddAbove s
    ⊢ BddAbove (Set.image f s)
  -/
  rw [bddAbove_iff_small] at hs ⊢
  /-
    f : Cardinal.{u} → Cardinal.{max u v}
    s : Set Cardinal.{u}
    hs : Small.{u, u + 1} ↑s
    ⊢ Small.{max u v, (max u v) + 1} ↑(Set.image f s)
  -/
  exact small_lift _
  /-
    🎉 no goals
  -/


theorem bddAbove_range_comp {ι : Type u} {f : ι → Cardinal.{v}} (hf : BddAbove (range f))
    (g : Cardinal.{v} → Cardinal.{max v w}) : BddAbove (range (g ∘ f)) := by
  /-
    ι : Type u
    f : ι → Cardinal.{v}
    hf : BddAbove (Set.range f)
    g : Cardinal.{v} → Cardinal.{max v w}
    ⊢ BddAbove (Set.range (Function.comp g f))
  -/
  rw [range_comp]
  /-
    ι : Type u
    f : ι → Cardinal.{v}
    hf : BddAbove (Set.range f)
    g : Cardinal.{v} → Cardinal.{max v w}
    ⊢ BddAbove (Set.image g (Set.range f))
  -/
  exact bddAbove_image g hf
  /-
    🎉 no goals
  -/


theorem sum_le_iSup_lift {ι : Type u}
    (f : ι → Cardinal.{max u v}) : sum f ≤ Cardinal.lift #ι * iSup f := by
  /-
    ι : Type u
    f : ι → Cardinal.{max u v}
    ⊢ LE.le (Cardinal.sum f) (HMul.hMul (Cardinal.lift.{v, u} (Cardinal.mk ι)) (iS …
  -/
  rw [← (iSup f).lift_id, ← lift_umax, lift_umax.{max u v, u}, ← sum_const]
  /-
    ι : Type u
    f : ι → Cardinal.{max u v}
    ⊢ LE.le (Cardinal.sum f) (Cardinal.sum fun x => iSup f)
  -/
  exact sum_le_sum _ _ (le_ciSup <| bddAbove_of_small _)
  /-
    🎉 no goals
  -/


theorem sum_le_iSup {ι : Type u} (f : ι → Cardinal.{u}) : sum f ≤ #ι * iSup f := by
  /-
    ι : Type u
    f : ι → Cardinal.{u}
    ⊢ LE.le (Cardinal.sum f) (HMul.hMul (Cardinal.mk ι) (iSup f))
  -/
  rw [← lift_id #ι]
  /-
    ι : Type u
    f : ι → Cardinal.{u}
    ⊢ LE.le (Cardinal.sum f) (HMul.hMul (Cardinal.lift.{u, u} (Cardinal.mk ι)) (iS …
  -/
  exact sum_le_iSup_lift f
  /-
    🎉 no goals
  -/


/-- The lift of a supremum is the supremum of the lifts. -/
theorem lift_sSup {s : Set Cardinal} (hs : BddAbove s) :
    lift.{u} (sSup s) = sSup (lift.{u} '' s) := by
  /-
    s : Set Cardinal.{u_1}
    hs : BddAbove s
    ⊢ Eq (Cardinal.lift.{u, u_1} (SupSet.sSup s)) (SupSet.sSup (Set.image Cardinal …
  -/
  apply ((le_csSup_iff' (bddAbove_image.{_,u} _ hs)).2 fun c hc => _).antisymm (csSup_le' _)
    /-
      s : Set Cardinal.{u_1}
      hs : BddAbove s
      ⊢ ∀ (c : Cardinal.{max u_1 u}), Membership.mem (upperBounds (Set.image Cardina …
    -/
  · intro c hc
    /-
      s : Set Cardinal.{u_1}
      hs : BddAbove s
      c : Cardinal.{max u_1 u}
      hc : Membership.mem (upperBounds (Set.image Cardinal.lift.{u, u_1} s)) c
      ⊢ LE.le (Cardinal.lift.{u, u_1} (SupSet.sSup s)) c
    -/
    by_contra h
    /-
      s : Set Cardinal.{u_1}
      hs : BddAbove s
      c : Cardinal.{max u_1 u}
      hc : Membership.mem (upperBounds (Set.image Cardinal.lift.{u, u_1} s)) c
      h : Not (LE.le (Cardinal.lift.{u, u_1} (SupSet.sSup s)) c)
      ⊢ False
    -/
    obtain ⟨d, rfl⟩ := Cardinal.mem_range_lift_of_le (not_le.1 h).le
    /-
      case intro
      s : Set Cardinal.{u_1}
      hs : BddAbove s
      d : Cardinal.{u_1}
      hc : Membership.mem (upperBounds (Set.image Cardinal.lift.{u, u_1} s)) (Cardin …
      h : Not (LE.le (Cardinal.lift.{u, u_1} (SupSet.sSup s)) (Cardinal.lift.{u, u_1 …
      ⊢ False
    -/
    simp_rw [lift_le] at h hc
    /-
      case intro
      s : Set Cardinal.{u_1}
      hs : BddAbove s
      d : Cardinal.{u_1}
      hc : Membership.mem (upperBounds (Set.image Cardinal.lift.{u, u_1} s)) (Cardin …
      h : Not (LE.le (SupSet.sSup s) d)
      ⊢ False
    -/
    rw [csSup_le_iff' hs] at h
    /-
      case intro
      s : Set Cardinal.{u_1}
      hs : BddAbove s
      d : Cardinal.{u_1}
      hc : Membership.mem (upperBounds (Set.image Cardinal.lift.{u, u_1} s)) (Cardin …
      h : Not (∀ (x : Cardinal.{u_1}), Membership.mem s x → LE.le x d)
      ⊢ False
    -/
    exact h fun a ha => lift_le.1 <| hc (mem_image_of_mem _ ha)
    /-
      🎉 no goals
    -/
    /-
      s : Set Cardinal.{u_1}
      hs : BddAbove s
      ⊢ Membership.mem (upperBounds (Set.image Cardinal.lift.{u, u_1} s)) (Cardinal. …
    -/
  · rintro i ⟨j, hj, rfl⟩
    /-
      case intro.intro
      s : Set Cardinal.{u_1}
      hs : BddAbove s
      j : Cardinal.{u_1}
      hj : Membership.mem s j
      ⊢ LE.le (Cardinal.lift.{u, u_1} j) (Cardinal.lift.{u, u_1} (SupSet.sSup s))
    -/
    exact lift_le.2 (le_csSup hs hj)
    /-
      🎉 no goals
    -/


/-- The lift of a supremum is the supremum of the lifts. -/
theorem lift_iSup {ι : Type v} {f : ι → Cardinal.{w}} (hf : BddAbove (range f)) :
    lift.{u} (iSup f) = ⨆ i, lift.{u} (f i) := by
  /-
    ι : Type v
    f : ι → Cardinal.{w}
    hf : BddAbove (Set.range f)
    ⊢ Eq (Cardinal.lift.{u, w} (iSup f)) (iSup fun i => Cardinal.lift.{u, w} (f i))
  -/
  rw [iSup, iSup, lift_sSup hf, ← range_comp]
  /-
    ι : Type v
    f : ι → Cardinal.{w}
    hf : BddAbove (Set.range f)
    ⊢ Eq (SupSet.sSup (Set.range (Function.comp Cardinal.lift.{u, w} f))) (SupSet. …
  -/
  simp [Function.comp_def]
  /-
    🎉 no goals
  -/


/-- To prove that the lift of a supremum is bounded by some cardinal `t`,
it suffices to show that the lift of each cardinal is bounded by `t`. -/
theorem lift_iSup_le {ι : Type v} {f : ι → Cardinal.{w}} {t : Cardinal} (hf : BddAbove (range f))
    (w : ∀ i, lift.{u} (f i) ≤ t) : lift.{u} (iSup f) ≤ t := by
  /-
    ι : Type v
    f : ι → Cardinal.{w}
    t : Cardinal.{max u w}
    hf : BddAbove (Set.range f)
    w : ∀ (i : ι), LE.le (Cardinal.lift.{u, w} (f i)) t
    ⊢ LE.le (Cardinal.lift.{u, w} (iSup f)) t
  -/
  rw [lift_iSup hf]
  /-
    ι : Type v
    f : ι → Cardinal.{w}
    t : Cardinal.{max u w}
    hf : BddAbove (Set.range f)
    w : ∀ (i : ι), LE.le (Cardinal.lift.{u, w} (f i)) t
    ⊢ LE.le (iSup fun i => Cardinal.lift.{u, w} (f i)) t
  -/
  exact ciSup_le' w
  /-
    🎉 no goals
  -/


@[simp]
theorem lift_iSup_le_iff {ι : Type v} {f : ι → Cardinal.{w}} (hf : BddAbove (range f))
    {t : Cardinal} : lift.{u} (iSup f) ≤ t ↔ ∀ i, lift.{u} (f i) ≤ t := by
  /-
    ι : Type v
    f : ι → Cardinal.{w}
    hf : BddAbove (Set.range f)
    t : Cardinal.{max u w}
    ⊢ Iff (LE.le (Cardinal.lift.{u, w} (iSup f)) t) (∀ (i : ι), LE.le (Cardinal.li …
  -/
  rw [lift_iSup hf]
  /-
    ι : Type v
    f : ι → Cardinal.{w}
    hf : BddAbove (Set.range f)
    t : Cardinal.{max u w}
    ⊢ Iff (LE.le (iSup fun i => Cardinal.lift.{u, w} (f i)) t) (∀ (i : ι), LE.le ( …
  -/
  exact ciSup_le_iff' (bddAbove_range_comp.{_,_,u} hf _)
  /-
    🎉 no goals
  -/


/-- To prove an inequality between the lifts to a common universe of two different supremums,
it suffices to show that the lift of each cardinal from the smaller supremum
if bounded by the lift of some cardinal from the larger supremum.
-/
theorem lift_iSup_le_lift_iSup {ι : Type v} {ι' : Type v'} {f : ι → Cardinal.{w}}
    {f' : ι' → Cardinal.{w'}} (hf : BddAbove (range f)) (hf' : BddAbove (range f')) {g : ι → ι'}
    (h : ∀ i, lift.{w'} (f i) ≤ lift.{w} (f' (g i))) : lift.{w'} (iSup f) ≤ lift.{w} (iSup f') := by
  /-
    ι : Type v
    ι' : Type v'
    f : ι → Cardinal.{w}
    f' : ι' → Cardinal.{w'}
    hf : BddAbove (Set.range f)
    hf' : BddAbove (Set.range f')
    g : ι → ι'
    h : ∀ (i : ι), LE.le (Cardinal.lift.{w', w} (f i)) (Cardinal.lift.{w, w'} (f'  …
    ⊢ LE.le (Cardinal.lift.{w', w} (iSup f)) (Cardinal.lift.{w, w'} (iSup f'))
  -/
  rw [lift_iSup hf, lift_iSup hf']
  /-
    ι : Type v
    ι' : Type v'
    f : ι → Cardinal.{w}
    f' : ι' → Cardinal.{w'}
    hf : BddAbove (Set.range f)
    hf' : BddAbove (Set.range f')
    g : ι → ι'
    h : ∀ (i : ι), LE.le (Cardinal.lift.{w', w} (f i)) (Cardinal.lift.{w, w'} (f'  …
    ⊢ LE.le (iSup fun i => Cardinal.lift.{w', w} (f i)) (iSup fun i => Cardinal.li …
  -/
  exact ciSup_mono' (bddAbove_range_comp.{_,_,w} hf' _) fun i => ⟨_, h i⟩
  /-
    🎉 no goals
  -/


/-- A variant of `lift_iSup_le_lift_iSup` with universes specialized via `w = v` and `w' = v'`.
This is sometimes necessary to avoid universe unification issues. -/
theorem lift_iSup_le_lift_iSup' {ι : Type v} {ι' : Type v'} {f : ι → Cardinal.{v}}
    {f' : ι' → Cardinal.{v'}} (hf : BddAbove (range f)) (hf' : BddAbove (range f')) (g : ι → ι')
    (h : ∀ i, lift.{v'} (f i) ≤ lift.{v} (f' (g i))) : lift.{v'} (iSup f) ≤ lift.{v} (iSup f') :=
  lift_iSup_le_lift_iSup hf hf' h


lemma exists_eq_of_iSup_eq_of_not_isSuccPrelimit
    {ι : Type u} (f : ι → Cardinal.{v}) (ω : Cardinal.{v})
    (hω : ¬ IsSuccPrelimit ω)
    (h : ⨆ i : ι, f i = ω) : ∃ i, f i = ω := by
  /-
    ι : Type u
    f : ι → Cardinal.{v}
    ω : Cardinal.{v}
    hω : Not (Order.IsSuccPrelimit ω)
    h : Eq (iSup fun i => f i) ω
    ⊢ Exists fun i => Eq (f i) ω
  -/
  subst h
  /-
    ι : Type u
    f : ι → Cardinal.{v}
    hω : Not (Order.IsSuccPrelimit (iSup fun i => f i))
    ⊢ Exists fun i => Eq (f i) (iSup fun i => f i)
  -/
  refine (isLUB_csSup' ?_).exists_of_not_isSuccPrelimit hω
  /-
    ι : Type u
    f : ι → Cardinal.{v}
    hω : Not (Order.IsSuccPrelimit (iSup fun i => f i))
    ⊢ BddAbove (Set.range f)
  -/
  contrapose! hω with hf
  /-
    ι : Type u
    f : ι → Cardinal.{v}
    hf : Not (BddAbove (Set.range f))
    ⊢ Order.IsSuccPrelimit (iSup fun i => f i)
  -/
  rw [iSup, csSup_of_not_bddAbove hf, csSup_empty]
  /-
    ι : Type u
    f : ι → Cardinal.{v}
    hf : Not (BddAbove (Set.range f))
    ⊢ Order.IsSuccPrelimit Bot.bot
  -/
  exact isSuccPrelimit_bot
  /-
    🎉 no goals
  -/


lemma exists_eq_of_iSup_eq_of_not_isSuccLimit
    {ι : Type u} [hι : Nonempty ι] (f : ι → Cardinal.{v}) (hf : BddAbove (range f))
    {c : Cardinal.{v}} (hc : ¬ IsSuccLimit c)
    (h : ⨆ i, f i = c) : ∃ i, f i = c := by
  /-
    ι : Type u
    hι : Nonempty ι
    f : ι → Cardinal.{v}
    hf : BddAbove (Set.range f)
    c : Cardinal.{v}
    hc : Not (Order.IsSuccLimit c)
    h : Eq (iSup fun i => f i) c
    ⊢ Exists fun i => Eq (f i) c
  -/
  rw [Cardinal.isSuccLimit_iff] at hc
  refine (not_and_or.mp hc).elim (fun e ↦ ⟨hι.some, ?_⟩)
    (Cardinal.exists_eq_of_iSup_eq_of_not_isSuccPrelimit.{u, v} f c · h)
  /-
    ι : Type u
    hι : Nonempty ι
    f : ι → Cardinal.{v}
    hf : BddAbove (Set.range f)
    c : Cardinal.{v}
    hc : Not (And (Ne c 0) (Order.IsSuccPrelimit c))
    h : Eq (iSup fun i => f i) c
    e : Not (Ne c 0)
    ⊢ Eq (f hι.some) c
  -/
  cases not_not.mp e
  /-
    case refl
    ι : Type u
    hι : Nonempty ι
    f : ι → Cardinal.{v}
    hf : BddAbove (Set.range f)
    hc : Not (And (Ne 0 0) (Order.IsSuccPrelimit 0))
    h : Eq (iSup fun i => f i) 0
    e : Not (Ne 0 0)
    ⊢ Eq (f hι.some) 0
  -/
  rw [← le_zero_iff] at h ⊢
  /-
    case refl
    ι : Type u
    hι : Nonempty ι
    f : ι → Cardinal.{v}
    hf : BddAbove (Set.range f)
    hc : Not (And (Ne 0 0) (Order.IsSuccPrelimit 0))
    h : LE.le (iSup fun i => f i) 0
    e : Not (Ne 0 0)
    ⊢ LE.le (f hι.some) 0
  -/
  exact (le_ciSup hf _).trans h
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated exists_eq_of_iSup_eq_of_not_isSuccLimit (since := "2024-09-17")]
lemma exists_eq_of_iSup_eq_of_not_isLimit
    {ι : Type u} [hι : Nonempty ι] (f : ι → Cardinal.{v}) (hf : BddAbove (range f))
    (ω : Cardinal.{v}) (hω : ¬ ω.IsLimit)
    (h : ⨆ i : ι, f i = ω) : ∃ i, f i = ω := by
  refine (not_and_or.mp hω).elim (fun e ↦ ⟨hι.some, ?_⟩)
    (Cardinal.exists_eq_of_iSup_eq_of_not_isSuccPrelimit.{u, v} f ω · h)
  /-
    ι : Type u
    hι : Nonempty ι
    f : ι → Cardinal.{v}
    hf : BddAbove (Set.range f)
    ω : Cardinal.{v}
    hω : Not ω.IsLimit
    h : Eq (iSup fun i => f i) ω
    e : Not (Ne ω 0)
    ⊢ Eq (f hι.some) ω
  -/
  cases not_not.mp e
  /-
    case refl
    ι : Type u
    hι : Nonempty ι
    f : ι → Cardinal.{v}
    hf : BddAbove (Set.range f)
    hω : Not (Cardinal.IsLimit 0)
    h : Eq (iSup fun i => f i) 0
    e : Not (Ne 0 0)
    ⊢ Eq (f hι.some) 0
  -/
  rw [← le_zero_iff] at h ⊢
  /-
    case refl
    ι : Type u
    hι : Nonempty ι
    f : ι → Cardinal.{v}
    hf : BddAbove (Set.range f)
    hω : Not (Cardinal.IsLimit 0)
    h : LE.le (iSup fun i => f i) 0
    e : Not (Ne 0 0)
    ⊢ LE.le (f hι.some) 0
  -/
  exact (le_ciSup hf _).trans h
  /-
    🎉 no goals
  -/


/-- The indexed product of cardinals is the cardinality of the Pi type
  (dependent product). -/
def prod {ι : Type u} (f : ι → Cardinal) : Cardinal :=
  #(Π i, (f i).out)


@[simp]
theorem mk_pi {ι : Type u} (α : ι → Type v) : #(Π i, α i) = prod fun i => #(α i) :=
  mk_congr <| Equiv.piCongrRight fun _ => outMkEquiv.symm


theorem mk_pi_congr_lift {ι : Type v} {ι' : Type v'} {f : ι → Type w} {g : ι' → Type w'}
    (e : ι ≃ ι') (h : ∀ i, lift.{w'} #(f i) = lift.{w} #(g (e i))) :
    lift.{max v' w'} #(Π i, f i) = lift.{max v w} #(Π i, g i) :=
  Cardinal.lift_mk_eq'.2 ⟨.piCongr e fun i ↦ Classical.choice <| Cardinal.lift_mk_eq'.1 (h i)⟩


theorem mk_pi_congr {ι ι' : Type u} {f : ι → Type v} {g : ι' → Type v} (e : ι ≃ ι')
    (h : ∀ i, #(f i) = #(g (e i))) : #(Π i, f i) = #(Π i, g i) :=
  mk_congr <| Equiv.piCongr e fun i ↦ Classical.choice <| Cardinal.eq.mp (h i)


theorem mk_pi_congr_prop {ι ι' : Prop} {f : ι → Type v} {g : ι' → Type v} (e : ι ↔ ι')
    (h : ∀ i, #(f i) = #(g (e.mp i))) : #(Π i, f i) = #(Π i, g i) :=
  mk_congr <| Equiv.piCongr (.ofIff e) fun i ↦ Classical.choice <| Cardinal.eq.mp (h i)


/-- Similar to `mk_pi_congr` with indexing types in different universes. This is not a strict
generalization. -/
theorem mk_pi_congr' {ι : Type u} {ι' : Type v} {f : ι → Type max w (max u v)}
    {g : ι' → Type max w (max u v)} (e : ι ≃ ι')
    (h : ∀ i, #(f i) = #(g (e i))) : #(Π i, f i) = #(Π i, g i) :=
  mk_congr <| Equiv.piCongr e fun i ↦ Classical.choice <| Cardinal.eq.mp (h i)


theorem mk_pi_congrRight {ι : Type u} {f g : ι → Type v} (h : ∀ i, #(f i) = #(g i)) :
    #(Π i, f i) = #(Π i, g i) :=
  mk_pi_congr (Equiv.refl ι) h


theorem mk_pi_congrRight_prop {ι : Prop} {f g : ι → Type v} (h : ∀ i, #(f i) = #(g i)) :
    #(Π i, f i) = #(Π i, g i) :=
  mk_pi_congr_prop Iff.rfl h


@[simp]
theorem prod_const (ι : Type u) (a : Cardinal.{v}) :
    (prod fun _ : ι => a) = lift.{u} a ^ lift.{v} #ι :=
  inductionOn a fun _ =>
    mk_congr <| Equiv.piCongr Equiv.ulift.symm fun _ => outMkEquiv.trans Equiv.ulift.symm


theorem prod_const' (ι : Type u) (a : Cardinal.{u}) : (prod fun _ : ι => a) = a ^ #ι :=
  inductionOn a fun _ => (mk_pi _).symm


theorem prod_le_prod {ι} (f g : ι → Cardinal) (H : ∀ i, f i ≤ g i) : prod f ≤ prod g :=
  ⟨Embedding.piCongrRight fun i =>
                             /-
                               ι : Type u_1
                               f g : ι → Cardinal.{u_2}
                               H : ∀ (i : ι), LE.le (f i) (g i)
                               i : ι
                               ⊢ Nonempty (Function.Embedding (Quotient.out (f i)) (Quotient.out (g i)))
                             -/
      Classical.choice <| by have := H i; rwa [← mk_out (f i), ← mk_out (g i)] at this⟩
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
theorem prod_eq_zero {ι} (f : ι → Cardinal.{u}) : prod f = 0 ↔ ∃ i, f i = 0 := by
  /-
    ι : Type u_1
    f : ι → Cardinal.{u}
    ⊢ Iff (Eq (Cardinal.prod f) 0) (Exists fun i => Eq (f i) 0)
  -/
  lift f to ι → Type u using fun _ => trivial
  /-
    case intro
    ι : Type u_1
    f : ι → Type u
    ⊢ Iff (Eq (Cardinal.prod fun i => Cardinal.mk (f i)) 0) (Exists fun i => Eq (( …
  -/
  simp only [mk_eq_zero_iff, ← mk_pi, isEmpty_pi]
  /-
    🎉 no goals
  -/


                                                                              /-
                                                                                ι : Type u_1
                                                                                f : ι → Cardinal.{u_2}
                                                                                ⊢ Iff (Ne (Cardinal.prod f) 0) (∀ (i : ι), Ne (f i) 0)
                                                                              -/
theorem prod_ne_zero {ι} (f : ι → Cardinal) : prod f ≠ 0 ↔ ∀ i, f i ≠ 0 := by simp [prod_eq_zero]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


theorem power_sum {ι} (a : Cardinal) (f : ι → Cardinal) :
    a ^ sum f = prod fun i ↦ a ^ f i := by
  induction a using Cardinal.inductionOn with | _ α =>
  induction f using induction_on_pi with | _ f =>
  simp_rw [prod, sum, power_def]
  apply mk_congr
  refine (Equiv.piCurry fun _ _ => α).trans ?_
  refine Equiv.piCongrRight fun b => ?_
  refine (Equiv.arrowCongr outMkEquiv (Equiv.refl α)).trans ?_
  exact outMkEquiv.symm


@[simp]
theorem lift_prod {ι : Type u} (c : ι → Cardinal.{v}) :
    lift.{w} (prod c) = prod fun i => lift.{w} (c i) := by
  /-
    ι : Type u
    c : ι → Cardinal.{v}
    ⊢ Eq (Cardinal.lift.{w, max v u} (Cardinal.prod c)) (Cardinal.prod fun i => Ca …
  -/
  lift c to ι → Type v using fun _ => trivial
  /-
    case intro
    ι : Type u
    c : ι → Type v
    ⊢ Eq (Cardinal.lift.{w, max v u} (Cardinal.prod fun i => Cardinal.mk (c i))) ( …
  -/
  simp only [← mk_pi, ← mk_uLift]
  /-
    case intro
    ι : Type u
    c : ι → Type v
    ⊢ Eq (Cardinal.mk (ULift.{w, max u v} ((i : ι) → c i))) (Cardinal.mk ((i : ι)  …
  -/
  exact mk_congr (Equiv.ulift.trans <| Equiv.piCongrRight fun i => Equiv.ulift.symm)
  /-
    🎉 no goals
  -/


theorem prod_eq_of_fintype {α : Type u} [h : Fintype α] (f : α → Cardinal.{v}) :
    prod f = Cardinal.lift.{u} (∏ i, f i) := by
  /-
    α : Type u
    h : Fintype α
    f : α → Cardinal.{v}
    ⊢ Eq (Cardinal.prod f) (Cardinal.lift.{u, v} (Finset.univ.prod fun i => f i))
  -/
  revert f
  /-
    α : Type u
    h : Fintype α
    ⊢ ∀ (f : α → Cardinal.{v}), Eq (Cardinal.prod f) (Cardinal.lift.{u, v} (Finset …
  -/
  refine Fintype.induction_empty_option ?_ ?_ ?_ α (h_fintype := h)
    /-
      case refine_1
      α : Type u
      h : Fintype α
      ⊢ ∀ (α β : Type u) [inst : Fintype β] (e : Equiv α β), (∀ (f : α → Cardinal.{v …
    -/
  · intro α β hβ e h f
    /-
      case refine_1
      α✝ : Type u
      h✝ : Fintype α✝
      α β : Type u
      hβ : Fintype β
      e : Equiv α β
      h : ∀ (f : α → Cardinal.{v}), Eq (Cardinal.prod f) (Cardinal.lift.{u, v} (Fins …
      f : β → Cardinal.{v}
      ⊢ Eq (Cardinal.prod f) (Cardinal.lift.{u, v} (Finset.univ.prod fun i => f i))
    -/
    letI := Fintype.ofEquiv β e.symm
    /-
      case refine_1
      α✝ : Type u
      h✝ : Fintype α✝
      α β : Type u
      hβ : Fintype β
      e : Equiv α β
      h : ∀ (f : α → Cardinal.{v}), Eq (Cardinal.prod f) (Cardinal.lift.{u, v} (Fins …
      f : β → Cardinal.{v}
      this : Fintype α := Fintype.ofEquiv β e.symm
      ⊢ Eq (Cardinal.prod f) (Cardinal.lift.{u, v} (Finset.univ.prod fun i => f i))
    -/
    rw [← e.prod_comp f, ← h]
    /-
      case refine_1
      α✝ : Type u
      h✝ : Fintype α✝
      α β : Type u
      hβ : Fintype β
      e : Equiv α β
      h : ∀ (f : α → Cardinal.{v}), Eq (Cardinal.prod f) (Cardinal.lift.{u, v} (Fins …
      f : β → Cardinal.{v}
      this : Fintype α := Fintype.ofEquiv β e.symm
      ⊢ Eq (Cardinal.prod f) (Cardinal.prod fun i => f (e i))
    -/
    exact mk_congr (e.piCongrLeft _).symm
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      h : Fintype α
      ⊢ ∀ (f : PEmpty.{u + 1} → Cardinal.{v}), Eq (Cardinal.prod f) (Cardinal.lift.{ …
    -/
  · intro f
    /-
      case refine_2
      α : Type u
      h : Fintype α
      f : PEmpty.{u + 1} → Cardinal.{v}
      ⊢ Eq (Cardinal.prod f) (Cardinal.lift.{u, v} (Finset.univ.prod fun i => f i))
    -/
    rw [Fintype.univ_pempty, Finset.prod_empty, lift_one, Cardinal.prod, mk_eq_one]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u
      h : Fintype α
      ⊢ ∀ (α : Type u) [inst : Fintype α], (∀ (f : α → Cardinal.{v}), Eq (Cardinal.p …
    -/
  · intro α hα h f
    rw [Cardinal.prod, mk_congr Equiv.piOptionEquivProd, mk_prod, lift_umax.{v, u}, mk_out, ←
        Cardinal.prod, lift_prod, Fintype.prod_option, lift_mul, ← h fun a => f (some a)]
    /-
      case refine_3
      α✝ : Type u
      h✝ : Fintype α✝
      α : Type u
      hα : Fintype α
      h : ∀ (f : α → Cardinal.{v}), Eq (Cardinal.prod f) (Cardinal.lift.{u, v} (Fins …
      f : Option α → Cardinal.{v}
      ⊢ Eq (HMul.hMul (Cardinal.lift.{u, v} (f Option.none)) (Cardinal.prod fun i => …
    -/
    simp only [lift_id]
    /-
      🎉 no goals
    -/


/-- **König's theorem** -/
theorem sum_lt_prod {ι} (f g : ι → Cardinal) (H : ∀ i, f i < g i) : sum f < prod g :=
  lt_of_not_ge fun ⟨F⟩ => by
    have : Inhabited (∀ i : ι, (g i).out) := by
      refine ⟨fun i => Classical.choice <| mk_ne_zero_iff.1 ?_⟩
      rw [mk_out]
      exact (H i).ne_bot
    /-
      ι : Type u_1
      f g : ι → Cardinal.{u_2}
      H : ∀ (i : ι), LT.lt (f i) (g i)
      x✝ : GE.ge (Cardinal.sum f) (Cardinal.prod g)
      F : Function.Embedding ((i : ι) → Quotient.out (g i)) (Sigma fun i => Quotient …
      this : Inhabited ((i : ι) → Quotient.out (g i))
      ⊢ False
    -/
    let G := invFun F
    /-
      ι : Type u_1
      f g : ι → Cardinal.{u_2}
      H : ∀ (i : ι), LT.lt (f i) (g i)
      x✝ : GE.ge (Cardinal.sum f) (Cardinal.prod g)
      F : Function.Embedding ((i : ι) → Quotient.out (g i)) (Sigma fun i => Quotient …
      this : Inhabited ((i : ι) → Quotient.out (g i))
      G : (Sigma fun i => Quotient.out (f i)) → (i : ι) → Quotient.out (g i) := Func …
      ⊢ False
    -/
    have sG : Surjective G := invFun_surjective F.2
    choose C hc using
      show ∀ i, ∃ b, ∀ a, G ⟨i, a⟩ i ≠ b by
        intro i
        simp only [not_exists.symm, not_forall.symm]
        refine fun h => (H i).not_le ?_
        rw [← mk_out (f i), ← mk_out (g i)]
        exact ⟨Embedding.ofSurjective _ h⟩
    /-
      ι : Type u_1
      f g : ι → Cardinal.{u_2}
      H : ∀ (i : ι), LT.lt (f i) (g i)
      x✝ : GE.ge (Cardinal.sum f) (Cardinal.prod g)
      F : Function.Embedding ((i : ι) → Quotient.out (g i)) (Sigma fun i => Quotient …
      this : Inhabited ((i : ι) → Quotient.out (g i))
      G : (Sigma fun i => Quotient.out (f i)) → (i : ι) → Quotient.out (g i) := Func …
      sG : Function.Surjective G
      C : (i : ι) → Quotient.out (g i)
      hc : ∀ (i : ι) (a : Quotient.out (f i)), Ne (G ⟨i, a⟩ i) (C i)
      ⊢ False
    -/
    let ⟨⟨i, a⟩, h⟩ := sG C
    /-
      ι : Type u_1
      f g : ι → Cardinal.{u_2}
      H : ∀ (i : ι), LT.lt (f i) (g i)
      x✝ : GE.ge (Cardinal.sum f) (Cardinal.prod g)
      F : Function.Embedding ((i : ι) → Quotient.out (g i)) (Sigma fun i => Quotient …
      this : Inhabited ((i : ι) → Quotient.out (g i))
      G : (Sigma fun i => Quotient.out (f i)) → (i : ι) → Quotient.out (g i) := Func …
      sG : Function.Surjective G
      C : (i : ι) → Quotient.out (g i)
      hc : ∀ (i : ι) (a : Quotient.out (f i)), Ne (G ⟨i, a⟩ i) (C i)
      i : ι
      a : Quotient.out (f i)
      h : Eq (G ⟨i, a⟩) C
      ⊢ False
    -/
    exact hc i a (congr_fun h _)
    /-
      🎉 no goals
    -/


/-- `ℵ₀` is the smallest infinite cardinal. -/
def aleph0 : Cardinal.{u} :=
  lift #ℕ


@[inherit_doc]
scoped notation "ℵ₀" => Cardinal.aleph0


theorem mk_nat : #ℕ = ℵ₀ :=
  (lift_id _).symm


theorem aleph0_ne_zero : ℵ₀ ≠ 0 :=
  mk_ne_zero _


theorem aleph0_pos : 0 < ℵ₀ :=
  pos_iff_ne_zero.2 aleph0_ne_zero


@[simp]
theorem lift_aleph0 : lift ℵ₀ = ℵ₀ :=
  lift_lift _


@[simp]
theorem aleph0_le_lift {c : Cardinal.{u}} : ℵ₀ ≤ lift.{v} c ↔ ℵ₀ ≤ c := by
  /-
    c : Cardinal.{u}
    ⊢ Iff (LE.le Cardinal.aleph0 (Cardinal.lift.{v, u} c)) (LE.le Cardinal.aleph0 c)
  -/
  simpa using lift_le (a := ℵ₀)
  /-
    🎉 no goals
  -/


@[simp]
theorem lift_le_aleph0 {c : Cardinal.{u}} : lift.{v} c ≤ ℵ₀ ↔ c ≤ ℵ₀ := by
  /-
    c : Cardinal.{u}
    ⊢ Iff (LE.le (Cardinal.lift.{v, u} c) Cardinal.aleph0) (LE.le c Cardinal.aleph0)
  -/
  simpa using lift_le (b := ℵ₀)
  /-
    🎉 no goals
  -/


@[simp]
theorem aleph0_lt_lift {c : Cardinal.{u}} : ℵ₀ < lift.{v} c ↔ ℵ₀ < c := by
  /-
    c : Cardinal.{u}
    ⊢ Iff (LT.lt Cardinal.aleph0 (Cardinal.lift.{v, u} c)) (LT.lt Cardinal.aleph0 c)
  -/
  simpa using lift_lt (a := ℵ₀)
  /-
    🎉 no goals
  -/


@[simp]
theorem lift_lt_aleph0 {c : Cardinal.{u}} : lift.{v} c < ℵ₀ ↔ c < ℵ₀ := by
  /-
    c : Cardinal.{u}
    ⊢ Iff (LT.lt (Cardinal.lift.{v, u} c) Cardinal.aleph0) (LT.lt c Cardinal.aleph0)
  -/
  simpa using lift_lt (b := ℵ₀)
  /-
    🎉 no goals
  -/


@[simp]
theorem aleph0_eq_lift {c : Cardinal.{u}} : ℵ₀ = lift.{v} c ↔ ℵ₀ = c := by
  /-
    c : Cardinal.{u}
    ⊢ Iff (Eq Cardinal.aleph0 (Cardinal.lift.{v, u} c)) (Eq Cardinal.aleph0 c)
  -/
  simpa using lift_inj (a := ℵ₀)
  /-
    🎉 no goals
  -/


@[simp]
theorem lift_eq_aleph0 {c : Cardinal.{u}} : lift.{v} c = ℵ₀ ↔ c = ℵ₀ := by
  /-
    c : Cardinal.{u}
    ⊢ Iff (Eq (Cardinal.lift.{v, u} c) Cardinal.aleph0) (Eq c Cardinal.aleph0)
  -/
  simpa using lift_inj (b := ℵ₀)
  /-
    🎉 no goals
  -/


                                            /-
                                              n : Nat
                                              ⊢ Eq (Cardinal.mk (Fin n)) ↑n
                                            -/
theorem mk_fin (n : ℕ) : #(Fin n) = n := by simp
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
                                                                     /-
                                                                       n : Nat
                                                                       ⊢ Eq (Cardinal.lift.{u, v} ↑n) ↑n
                                                                     -/
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
theorem lift_natCast (n : ℕ) : lift.{u} (n : Cardinal.{v}) = n := by induction n <;> simp [*]
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem lift_ofNat (n : ℕ) [n.AtLeastTwo] :
    lift.{u} (no_index (OfNat.ofNat n : Cardinal.{v})) = OfNat.ofNat n :=
  lift_natCast n


@[simp]
theorem lift_eq_nat_iff {a : Cardinal.{u}} {n : ℕ} : lift.{v} a = n ↔ a = n :=
  lift_injective.eq_iff' (lift_natCast n)


@[simp]
theorem lift_eq_ofNat_iff {a : Cardinal.{u}} {n : ℕ} [n.AtLeastTwo] :
    lift.{v} a = (no_index (OfNat.ofNat n)) ↔ a = OfNat.ofNat n :=
  lift_eq_nat_iff


@[simp]
theorem nat_eq_lift_iff {n : ℕ} {a : Cardinal.{u}} :
    (n : Cardinal) = lift.{v} a ↔ (n : Cardinal) = a := by
  /-
    n : Nat
    a : Cardinal.{u}
    ⊢ Iff (Eq (↑n) (Cardinal.lift.{v, u} a)) (Eq (↑n) a)
  -/
  rw [← lift_natCast.{v,u} n, lift_inj]
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_eq_lift_iff {a : Cardinal.{u}} :
    (0 : Cardinal) = lift.{v} a ↔ 0 = a := by
  /-
    a : Cardinal.{u}
    ⊢ Iff (Eq 0 (Cardinal.lift.{v, u} a)) (Eq 0 a)
  -/
  simpa using nat_eq_lift_iff (n := 0)
  /-
    🎉 no goals
  -/


@[simp]
theorem one_eq_lift_iff {a : Cardinal.{u}} :
    (1 : Cardinal) = lift.{v} a ↔ 1 = a := by
  /-
    a : Cardinal.{u}
    ⊢ Iff (Eq 1 (Cardinal.lift.{v, u} a)) (Eq 1 a)
  -/
  simpa using nat_eq_lift_iff (n := 1)
  /-
    🎉 no goals
  -/

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem ofNat_eq_lift_iff {a : Cardinal.{u}} {n : ℕ} [n.AtLeastTwo] :
    (no_index (OfNat.ofNat n : Cardinal)) = lift.{v} a ↔ (OfNat.ofNat n : Cardinal) = a :=
  nat_eq_lift_iff


@[simp]
theorem lift_le_nat_iff {a : Cardinal.{u}} {n : ℕ} : lift.{v} a ≤ n ↔ a ≤ n := by
  /-
    a : Cardinal.{u}
    n : Nat
    ⊢ Iff (LE.le (Cardinal.lift.{v, u} a) ↑n) (LE.le a ↑n)
  -/
  rw [← lift_natCast.{v,u}, lift_le]
  /-
    🎉 no goals
  -/


@[simp]
theorem lift_le_one_iff {a : Cardinal.{u}} :
    lift.{v} a ≤ 1 ↔ a ≤ 1 := by
  /-
    a : Cardinal.{u}
    ⊢ Iff (LE.le (Cardinal.lift.{v, u} a) 1) (LE.le a 1)
  -/
  simpa using lift_le_nat_iff (n := 1)
  /-
    🎉 no goals
  -/

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem lift_le_ofNat_iff {a : Cardinal.{u}} {n : ℕ} [n.AtLeastTwo] :
    lift.{v} a ≤ (no_index (OfNat.ofNat n)) ↔ a ≤ OfNat.ofNat n :=
  lift_le_nat_iff


@[simp]
theorem nat_le_lift_iff {n : ℕ} {a : Cardinal.{u}} : n ≤ lift.{v} a ↔ n ≤ a := by
  /-
    n : Nat
    a : Cardinal.{u}
    ⊢ Iff (LE.le (↑n) (Cardinal.lift.{v, u} a)) (LE.le (↑n) a)
  -/
  rw [← lift_natCast.{v,u}, lift_le]
  /-
    🎉 no goals
  -/


@[simp]
theorem one_le_lift_iff {a : Cardinal.{u}} :
    (1 : Cardinal) ≤ lift.{v} a ↔ 1 ≤ a := by
  /-
    a : Cardinal.{u}
    ⊢ Iff (LE.le 1 (Cardinal.lift.{v, u} a)) (LE.le 1 a)
  -/
  simpa using nat_le_lift_iff (n := 1)
  /-
    🎉 no goals
  -/

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem ofNat_le_lift_iff {a : Cardinal.{u}} {n : ℕ} [n.AtLeastTwo] :
    (no_index (OfNat.ofNat n : Cardinal)) ≤ lift.{v} a ↔ (OfNat.ofNat n : Cardinal) ≤ a :=
  nat_le_lift_iff


@[simp]
theorem lift_lt_nat_iff {a : Cardinal.{u}} {n : ℕ} : lift.{v} a < n ↔ a < n := by
  /-
    a : Cardinal.{u}
    n : Nat
    ⊢ Iff (LT.lt (Cardinal.lift.{v, u} a) ↑n) (LT.lt a ↑n)
  -/
  rw [← lift_natCast.{v,u}, lift_lt]
  /-
    🎉 no goals
  -/

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem lift_lt_ofNat_iff {a : Cardinal.{u}} {n : ℕ} [n.AtLeastTwo] :
    lift.{v} a < (no_index (OfNat.ofNat n)) ↔ a < OfNat.ofNat n :=
  lift_lt_nat_iff


@[simp]
theorem nat_lt_lift_iff {n : ℕ} {a : Cardinal.{u}} : n < lift.{v} a ↔ n < a := by
  /-
    n : Nat
    a : Cardinal.{u}
    ⊢ Iff (LT.lt (↑n) (Cardinal.lift.{v, u} a)) (LT.lt (↑n) a)
  -/
  rw [← lift_natCast.{v,u}, lift_lt]
  /-
    🎉 no goals
  -/

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem zero_lt_lift_iff {a : Cardinal.{u}} :
    (0 : Cardinal) < lift.{v} a ↔ 0 < a := by
  /-
    a : Cardinal.{u}
    ⊢ Iff (LT.lt 0 (Cardinal.lift.{v, u} a)) (LT.lt 0 a)
  -/
  simpa using nat_lt_lift_iff (n := 0)
  /-
    🎉 no goals
  -/


@[simp]
theorem one_lt_lift_iff {a : Cardinal.{u}} :
    (1 : Cardinal) < lift.{v} a ↔ 1 < a := by
  /-
    a : Cardinal.{u}
    ⊢ Iff (LT.lt 1 (Cardinal.lift.{v, u} a)) (LT.lt 1 a)
  -/
  simpa using nat_lt_lift_iff (n := 1)
  /-
    🎉 no goals
  -/

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem ofNat_lt_lift_iff {a : Cardinal.{u}} {n : ℕ} [n.AtLeastTwo] :
    (no_index (OfNat.ofNat n : Cardinal)) < lift.{v} a ↔ (OfNat.ofNat n : Cardinal) < a :=
  nat_lt_lift_iff


theorem lift_mk_fin (n : ℕ) : lift #(Fin n) = n := rfl


                                                                                /-
                                                                                  α : Type u
                                                                                  s : Finset α
                                                                                  ⊢ Eq (Cardinal.mk (Subtype fun x => Membership.mem s x)) ↑s.card
                                                                                -/
theorem mk_coe_finset {α : Type u} {s : Finset α} : #s = ↑(Finset.card s) := by simp
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


theorem mk_finset_of_fintype [Fintype α] : #(Finset α) = 2 ^ Fintype.card α := by
  /-
    α : Type u
    inst✝ : Fintype α
    ⊢ Eq (Cardinal.mk (Finset α)) (HPow.hPow 2 (Fintype.card α))
  -/
  simp [Pow.pow]
  /-
    🎉 no goals
  -/


theorem card_le_of_finset {α} (s : Finset α) : (s.card : Cardinal) ≤ #α :=
  @mk_coe_finset _ s ▸ mk_set_le _


instance : CharZero Cardinal := by
  /-
    α β : Type u
    ⊢ CharZero Cardinal.{u_1}
  -/
  refine ⟨fun a b h ↦ ?_⟩
  rwa [← lift_mk_fin, ← lift_mk_fin, lift_inj, Cardinal.eq, ← Fintype.card_eq,
    Fintype.card_fin, Fintype.card_fin] at h


@[deprecated Nat.cast_le (since := "2024-10-16")]
theorem natCast_le {m n : ℕ} : (m : Cardinal) ≤ n ↔ m ≤ n := Nat.cast_le


@[deprecated Nat.cast_lt (since := "2024-10-16")]
theorem natCast_lt {m n : ℕ} : (m : Cardinal) < n ↔ m < n := Nat.cast_lt


@[deprecated Nat.cast_inj (since := "2024-10-16")]
theorem natCast_inj {m n : ℕ} : (m : Cardinal) = n ↔ m = n := Nat.cast_inj


@[deprecated Nat.cast_injective (since := "2024-10-16")]
theorem natCast_injective : Injective ((↑) : ℕ → Cardinal) := Nat.cast_injective


@[deprecated Nat.cast_pow (since := "2024-10-16")]
theorem natCast_pow {m n : ℕ} : (↑(m ^ n) : Cardinal) = (↑m : Cardinal) ^ (↑n : Cardinal) :=
  Nat.cast_pow m n


@[norm_cast]
theorem nat_succ (n : ℕ) : (n.succ : Cardinal) = succ ↑n := by
  /-
    n : Nat
    ⊢ Eq (↑n.succ) (Order.succ ↑n)
  -/
  rw [Nat.cast_succ]
  /-
    n : Nat
    ⊢ Eq (HAdd.hAdd (↑n) 1) (Order.succ ↑n)
  -/
  refine (add_one_le_succ _).antisymm (succ_le_of_lt ?_)
  /-
    n : Nat
    ⊢ LT.lt (↑n) (HAdd.hAdd (↑n) 1)
  -/
  rw [← Nat.cast_succ]
  /-
    n : Nat
    ⊢ LT.lt ↑n ↑n.succ
  -/
  exact Nat.cast_lt.2 (Nat.lt_succ_self _)
  /-
    🎉 no goals
  -/


lemma succ_natCast (n : ℕ) : Order.succ (n : Cardinal) = n + 1 := by
  /-
    n : Nat
    ⊢ Eq (Order.succ ↑n) (HAdd.hAdd (↑n) 1)
  -/
  rw [← Cardinal.nat_succ]
  /-
    n : Nat
    ⊢ Eq (↑n.succ) (HAdd.hAdd (↑n) 1)
  -/
  norm_cast
  /-
    🎉 no goals
  -/


lemma natCast_add_one_le_iff {n : ℕ} {c : Cardinal} : n + 1 ≤ c ↔ n < c := by
  /-
    n : Nat
    c : Cardinal.{u_1}
    ⊢ Iff (LE.le (HAdd.hAdd (↑n) 1) c) (LT.lt (↑n) c)
  -/
  rw [← Order.succ_le_iff, Cardinal.succ_natCast]
  /-
    🎉 no goals
  -/


lemma two_le_iff_one_lt {c : Cardinal} : 2 ≤ c ↔ 1 < c := by
  /-
    c : Cardinal.{u_1}
    ⊢ Iff (LE.le 2 c) (LT.lt 1 c)
  -/
  convert natCast_add_one_le_iff
  /-
    case h.e'_1.h.e'_3
    c : Cardinal.{u_1}
    ⊢ Eq 2 (HAdd.hAdd (↑1) 1)
  -/
  norm_cast
  /-
    🎉 no goals
  -/


@[simp]
                                                  /-
                                                    ⊢ Eq (Order.succ 0) 1
                                                  -/
theorem succ_zero : succ (0 : Cardinal) = 1 := by norm_cast
                                                  /-
                                                    🎉 no goals
                                                  -/

-- This works generally to prove inequalities between numeric cardinals.

                                              /-
                                                ⊢ LT.lt 1 2
                                              -/
theorem one_lt_two : (1 : Cardinal) < 2 := by norm_cast
                                              /-
                                                🎉 no goals
                                              -/


theorem exists_finset_le_card (α : Type*) (n : ℕ) (h : n ≤ #α) :
    ∃ s : Finset α, n ≤ s.card := by
  /-
    α : Type u_1
    n : Nat
    h : LE.le (↑n) (Cardinal.mk α)
    ⊢ Exists fun s => LE.le n s.card
  -/
  obtain hα|hα := finite_or_infinite α
    /-
      case inl
      α : Type u_1
      n : Nat
      h : LE.le (↑n) (Cardinal.mk α)
      hα : Finite α
      ⊢ Exists fun s => LE.le n s.card
    -/
  · let hα := Fintype.ofFinite α
    /-
      case inl
      α : Type u_1
      n : Nat
      h : LE.le (↑n) (Cardinal.mk α)
      hα✝ : Finite α
      hα : Fintype α := Fintype.ofFinite α
      ⊢ Exists fun s => LE.le n s.card
    -/
    use Finset.univ
    /-
      case h
      α : Type u_1
      n : Nat
      h : LE.le (↑n) (Cardinal.mk α)
      hα✝ : Finite α
      hα : Fintype α := Fintype.ofFinite α
      ⊢ LE.le n Finset.univ.card
    -/
    simpa only [mk_fintype, Nat.cast_le] using h
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      n : Nat
      h : LE.le (↑n) (Cardinal.mk α)
      hα : Infinite α
      ⊢ Exists fun s => LE.le n s.card
    -/
  · obtain ⟨s, hs⟩ := Infinite.exists_subset_card_eq α n
    /-
      case inr.intro
      α : Type u_1
      n : Nat
      h : LE.le (↑n) (Cardinal.mk α)
      hα : Infinite α
      s : Finset α
      hs : Eq s.card n
      ⊢ Exists fun s => LE.le n s.card
    -/
    exact ⟨s, hs.ge⟩
    /-
      🎉 no goals
    -/


theorem card_le_of {α : Type u} {n : ℕ} (H : ∀ s : Finset α, s.card ≤ n) : #α ≤ n := by
  /-
    α : Type u
    n : Nat
    H : ∀ (s : Finset α), LE.le s.card n
    ⊢ LE.le (Cardinal.mk α) ↑n
  -/
  contrapose! H
  /-
    α : Type u
    n : Nat
    H : LT.lt (↑n) (Cardinal.mk α)
    ⊢ Exists fun s => LT.lt n s.card
  -/
  apply exists_finset_le_card α (n+1)
  /-
    α : Type u
    n : Nat
    H : LT.lt (↑n) (Cardinal.mk α)
    ⊢ LE.le (↑(HAdd.hAdd n 1)) (Cardinal.mk α)
  -/
  simpa only [nat_succ, succ_le_iff] using H
  /-
    🎉 no goals
  -/


theorem cantor' (a) {b : Cardinal} (hb : 1 < b) : a < b ^ a := by
  /-
    a b : Cardinal.{u_1}
    hb : LT.lt 1 b
    ⊢ LT.lt a (HPow.hPow b a)
  -/
  rw [← succ_le_iff, (by norm_cast : succ (1 : Cardinal) = 2)] at hb
  /-
    a b : Cardinal.{u_1}
    hb : LE.le 2 b
    ⊢ LT.lt a (HPow.hPow b a)
  -/
  exact (cantor a).trans_le (power_le_power_right hb)
  /-
    🎉 no goals
  -/


theorem one_le_iff_pos {c : Cardinal} : 1 ≤ c ↔ 0 < c := by
  /-
    c : Cardinal.{u_1}
    ⊢ Iff (LE.le 1 c) (LT.lt 0 c)
  -/
  rw [← succ_zero, succ_le_iff]
  /-
    🎉 no goals
  -/


theorem one_le_iff_ne_zero {c : Cardinal} : 1 ≤ c ↔ c ≠ 0 := by
  /-
    c : Cardinal.{u_1}
    ⊢ Iff (LE.le 1 c) (Ne c 0)
  -/
  rw [one_le_iff_pos, pos_iff_ne_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem lt_one_iff_zero {c : Cardinal} : c < 1 ↔ c = 0 := by
  /-
    c : Cardinal.{u_1}
    ⊢ Iff (LT.lt c 1) (Eq c 0)
  -/
  simpa using lt_succ_bot_iff (a := c)
  /-
    🎉 no goals
  -/


theorem nat_lt_aleph0 (n : ℕ) : (n : Cardinal.{u}) < ℵ₀ :=
  succ_le_iff.1
    (by
      /-
        n : Nat
        ⊢ LE.le (Order.succ ↑n) Cardinal.aleph0
      -/
      rw [← nat_succ, ← lift_mk_fin, aleph0, lift_mk_le.{u}]
      /-
        n : Nat
        ⊢ Nonempty (Function.Embedding (Fin n.succ) Nat)
      -/
      exact ⟨⟨(↑), fun a b => Fin.ext⟩⟩)
      /-
        🎉 no goals
      -/


@[simp]
                                     /-
                                       ⊢ LT.lt 1 Cardinal.aleph0
                                     -/
theorem one_lt_aleph0 : 1 < ℵ₀ := by simpa using nat_lt_aleph0 1
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem one_le_aleph0 : 1 ≤ ℵ₀ :=
  one_lt_aleph0.le


theorem lt_aleph0 {c : Cardinal} : c < ℵ₀ ↔ ∃ n : ℕ, c = n :=
  ⟨fun h => by
    /-
      c : Cardinal.{u_1}
      h : LT.lt c Cardinal.aleph0
      ⊢ Exists fun n => Eq c ↑n
    -/
    rcases lt_lift_iff.1 h with ⟨c, h', rfl⟩
    /-
      case intro.intro
      c : Cardinal.{0}
      h' : LT.lt c (Cardinal.mk Nat)
      h : LT.lt (Cardinal.lift.{u_1, 0} c) Cardinal.aleph0
      ⊢ Exists fun n => Eq (Cardinal.lift.{u_1, 0} c) ↑n
    -/
    rcases le_mk_iff_exists_set.1 h'.1 with ⟨S, rfl⟩
    suffices S.Finite by
      lift S to Finset ℕ using this
      simp
    /-
      case intro.intro.intro
      S : Set Nat
      h' : LT.lt (Cardinal.mk ↑S) (Cardinal.mk Nat)
      h : LT.lt (Cardinal.lift.{u_1, 0} (Cardinal.mk ↑S)) Cardinal.aleph0
      ⊢ S.Finite
    -/
    contrapose! h'
    /-
      case intro.intro.intro
      S : Set Nat
      h : LT.lt (Cardinal.lift.{u_1, 0} (Cardinal.mk ↑S)) Cardinal.aleph0
      h' : Not S.Finite
      ⊢ LE.le (Cardinal.mk Nat) (Cardinal.mk ↑S)
    -/
    haveI := Infinite.to_subtype h'
    /-
      case intro.intro.intro
      S : Set Nat
      h : LT.lt (Cardinal.lift.{u_1, 0} (Cardinal.mk ↑S)) Cardinal.aleph0
      h' : Not S.Finite
      this : Infinite ↑S
      ⊢ LE.le (Cardinal.mk Nat) (Cardinal.mk ↑S)
    -/
    exact ⟨Infinite.natEmbedding S⟩, fun ⟨_, e⟩ => e.symm ▸ nat_lt_aleph0 _⟩
    /-
      🎉 no goals
    -/


lemma succ_eq_of_lt_aleph0 {c : Cardinal} (h : c < ℵ₀) : Order.succ c = c + 1 := by
  /-
    c : Cardinal.{u_1}
    h : LT.lt c Cardinal.aleph0
    ⊢ Eq (Order.succ c) (HAdd.hAdd c 1)
  -/
  obtain ⟨n, hn⟩ := Cardinal.lt_aleph0.mp h
  /-
    case intro
    c : Cardinal.{u_1}
    h : LT.lt c Cardinal.aleph0
    n : Nat
    hn : Eq c ↑n
    ⊢ Eq (Order.succ c) (HAdd.hAdd c 1)
  -/
  rw [hn, succ_natCast]
  /-
    🎉 no goals
  -/


theorem aleph0_le {c : Cardinal} : ℵ₀ ≤ c ↔ ∀ n : ℕ, ↑n ≤ c :=
  ⟨fun h _ => (nat_lt_aleph0 _).le.trans h, fun h =>
    le_of_not_lt fun hn => by
      /-
        c : Cardinal.{u_1}
        h : ∀ (n : Nat), LE.le (↑n) c
        hn : LT.lt c Cardinal.aleph0
        ⊢ False
      -/
      rcases lt_aleph0.1 hn with ⟨n, rfl⟩
      /-
        case intro
        n : Nat
        h : ∀ (n_1 : Nat), LE.le ↑n_1 ↑n
        hn : LT.lt (↑n) Cardinal.aleph0
        ⊢ False
      -/
      exact (Nat.lt_succ_self _).not_le (Nat.cast_le.1 (h (n + 1)))⟩
      /-
        🎉 no goals
      -/


theorem isSuccPrelimit_aleph0 : IsSuccPrelimit ℵ₀ :=
  isSuccPrelimit_of_succ_lt fun a ha => by
    /-
      a : Cardinal.{u_1}
      ha : LT.lt a Cardinal.aleph0
      ⊢ LT.lt (Order.succ a) Cardinal.aleph0
    -/
    rcases lt_aleph0.1 ha with ⟨n, rfl⟩
    /-
      case intro
      n : Nat
      ha : LT.lt (↑n) Cardinal.aleph0
      ⊢ LT.lt (Order.succ ↑n) Cardinal.aleph0
    -/
    rw [← nat_succ]
    /-
      case intro
      n : Nat
      ha : LT.lt (↑n) Cardinal.aleph0
      ⊢ LT.lt (↑n.succ) Cardinal.aleph0
    -/
    apply nat_lt_aleph0
    /-
      🎉 no goals
    -/


theorem isSuccLimit_aleph0 : IsSuccLimit ℵ₀ := by
  /-
    ⊢ Order.IsSuccLimit Cardinal.aleph0
  -/
  rw [Cardinal.isSuccLimit_iff]
  /-
    ⊢ And (Ne Cardinal.aleph0 0) (Order.IsSuccPrelimit Cardinal.aleph0)
  -/
  exact ⟨aleph0_ne_zero, isSuccPrelimit_aleph0⟩
  /-
    🎉 no goals
  -/


lemma not_isSuccLimit_natCast : (n : ℕ) → ¬ IsSuccLimit (n : Cardinal.{u})
  | 0, e => e.1 isMin_bot
  | Nat.succ n, e => Order.not_isSuccPrelimit_succ _ (nat_succ n ▸ e.2)


theorem not_isSuccLimit_of_lt_aleph0 {c : Cardinal} (h : c < ℵ₀) : ¬ IsSuccLimit c := by
  /-
    c : Cardinal.{u_1}
    h : LT.lt c Cardinal.aleph0
    ⊢ Not (Order.IsSuccLimit c)
  -/
  obtain ⟨n, rfl⟩ := lt_aleph0.1 h
  /-
    case intro
    n : Nat
    h : LT.lt (↑n) Cardinal.aleph0
    ⊢ Not (Order.IsSuccLimit ↑n)
  -/
  exact not_isSuccLimit_natCast n
  /-
    🎉 no goals
  -/


theorem aleph0_le_of_isSuccLimit {c : Cardinal} (h : IsSuccLimit c) : ℵ₀ ≤ c := by
  /-
    c : Cardinal.{u_1}
    h : Order.IsSuccLimit c
    ⊢ LE.le Cardinal.aleph0 c
  -/
  contrapose! h
  /-
    c : Cardinal.{u_1}
    h : LT.lt c Cardinal.aleph0
    ⊢ Not (Order.IsSuccLimit c)
  -/
  exact not_isSuccLimit_of_lt_aleph0 h
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated isSuccLimit_aleph0 (since := "2024-09-17")]
theorem isLimit_aleph0 : IsLimit ℵ₀ :=
  ⟨aleph0_ne_zero, isSuccPrelimit_aleph0⟩


set_option linter.deprecated false in
@[deprecated not_isSuccLimit_natCast (since := "2024-09-17")]
lemma not_isLimit_natCast : (n : ℕ) → ¬ IsLimit (n : Cardinal.{u})
  | 0, e => e.1 rfl
  | Nat.succ n, e => Order.not_isSuccPrelimit_succ _ (nat_succ n ▸ e.2)


set_option linter.deprecated false in
@[deprecated aleph0_le_of_isSuccLimit (since := "2024-09-17")]
theorem IsLimit.aleph0_le {c : Cardinal} (h : IsLimit c) : ℵ₀ ≤ c := by
  /-
    c : Cardinal.{u_1}
    h : c.IsLimit
    ⊢ LE.le Cardinal.aleph0 c
  -/
  by_contra! h'
  /-
    c : Cardinal.{u_1}
    h : c.IsLimit
    h' : LT.lt c Cardinal.aleph0
    ⊢ False
  -/
  rcases lt_aleph0.1 h' with ⟨n, rfl⟩
  /-
    case intro
    n : Nat
    h : (↑n).IsLimit
    h' : LT.lt (↑n) Cardinal.aleph0
    ⊢ False
  -/
  exact not_isLimit_natCast n h
  /-
    🎉 no goals
  -/


lemma exists_eq_natCast_of_iSup_eq {ι : Type u} [Nonempty ι] (f : ι → Cardinal.{v})
    (hf : BddAbove (range f)) (n : ℕ) (h : ⨆ i, f i = n) : ∃ i, f i = n :=
  exists_eq_of_iSup_eq_of_not_isSuccLimit.{u, v} f hf (not_isSuccLimit_natCast n) h


@[simp]
theorem range_natCast : range ((↑) : ℕ → Cardinal) = Iio ℵ₀ :=
                  /-
                    x : Cardinal.{u_1}
                    ⊢ Iff (Membership.mem (Set.range Nat.cast) x) (Membership.mem (Set.Iio Cardina …
                  -/
  ext fun x => by simp only [mem_Iio, mem_range, eq_comm, lt_aleph0]
                  /-
                    🎉 no goals
                  -/


theorem mk_eq_nat_iff {α : Type u} {n : ℕ} : #α = n ↔ Nonempty (α ≃ Fin n) := by
  /-
    α : Type u
    n : Nat
    ⊢ Iff (Eq (Cardinal.mk α) ↑n) (Nonempty (Equiv α (Fin n)))
  -/
  rw [← lift_mk_fin, ← lift_uzero #α, lift_mk_eq']
  /-
    🎉 no goals
  -/


theorem lt_aleph0_iff_finite {α : Type u} : #α < ℵ₀ ↔ Finite α := by
  /-
    α : Type u
    ⊢ Iff (LT.lt (Cardinal.mk α) Cardinal.aleph0) (Finite α)
  -/
  simp only [lt_aleph0, mk_eq_nat_iff, finite_iff_exists_equiv_fin]
  /-
    🎉 no goals
  -/


theorem lt_aleph0_iff_fintype {α : Type u} : #α < ℵ₀ ↔ Nonempty (Fintype α) :=
  lt_aleph0_iff_finite.trans (finite_iff_nonempty_fintype _)


theorem lt_aleph0_of_finite (α : Type u) [Finite α] : #α < ℵ₀ :=
  lt_aleph0_iff_finite.2 ‹_›


theorem lt_aleph0_iff_set_finite {S : Set α} : #S < ℵ₀ ↔ S.Finite :=
  lt_aleph0_iff_finite.trans finite_coe_iff


alias ⟨_, _root_.Set.Finite.lt_aleph0⟩ := lt_aleph0_iff_set_finite


@[simp]
theorem lt_aleph0_iff_subtype_finite {p : α → Prop} : #{ x // p x } < ℵ₀ ↔ { x | p x }.Finite :=
  lt_aleph0_iff_set_finite


theorem mk_le_aleph0_iff : #α ≤ ℵ₀ ↔ Countable α := by
  /-
    α : Type u
    ⊢ Iff (LE.le (Cardinal.mk α) Cardinal.aleph0) (Countable α)
  -/
  rw [countable_iff_nonempty_embedding, aleph0, ← lift_uzero #α, lift_mk_le']
  /-
    🎉 no goals
  -/


@[simp]
theorem mk_le_aleph0 [Countable α] : #α ≤ ℵ₀ :=
  mk_le_aleph0_iff.mpr ‹_›


theorem le_aleph0_iff_set_countable {s : Set α} : #s ≤ ℵ₀ ↔ s.Countable := mk_le_aleph0_iff


alias ⟨_, _root_.Set.Countable.le_aleph0⟩ := le_aleph0_iff_set_countable


@[simp]
theorem le_aleph0_iff_subtype_countable {p : α → Prop} :
    #{ x // p x } ≤ ℵ₀ ↔ { x | p x }.Countable :=
  le_aleph0_iff_set_countable


theorem aleph0_lt_mk_iff : ℵ₀ < #α ↔ Uncountable α := by
  /-
    α : Type u
    ⊢ Iff (LT.lt Cardinal.aleph0 (Cardinal.mk α)) (Uncountable α)
  -/
  rw [← not_le, ← not_countable_iff, not_iff_not, mk_le_aleph0_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem aleph0_lt_mk [Uncountable α] : ℵ₀ < #α :=
  aleph0_lt_mk_iff.mpr ‹_›


instance canLiftCardinalNat : CanLift Cardinal ℕ (↑) fun x => x < ℵ₀ :=
  ⟨fun _ hx =>
    let ⟨n, hn⟩ := lt_aleph0.mp hx
    ⟨n, hn.symm⟩⟩


theorem add_lt_aleph0 {a b : Cardinal} (ha : a < ℵ₀) (hb : b < ℵ₀) : a + b < ℵ₀ :=
  match a, b, lt_aleph0.1 ha, lt_aleph0.1 hb with
                                   /-
                                     a b : Cardinal.{u_1}
                                     m n : Nat
                                     ha : LT.lt (↑m) Cardinal.aleph0
                                     hb : LT.lt (↑n) Cardinal.aleph0
                                     ⊢ LT.lt (HAdd.hAdd ↑m ↑n) Cardinal.aleph0
                                   -/
  | _, _, ⟨m, rfl⟩, ⟨n, rfl⟩ => by rw [← Nat.cast_add]; apply nat_lt_aleph0
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem add_lt_aleph0_iff {a b : Cardinal} : a + b < ℵ₀ ↔ a < ℵ₀ ∧ b < ℵ₀ :=
  ⟨fun h => ⟨(self_le_add_right _ _).trans_lt h, (self_le_add_left _ _).trans_lt h⟩,
   fun ⟨h1, h2⟩ => add_lt_aleph0 h1 h2⟩


theorem aleph0_le_add_iff {a b : Cardinal} : ℵ₀ ≤ a + b ↔ ℵ₀ ≤ a ∨ ℵ₀ ≤ b := by
  /-
    a b : Cardinal.{u_1}
    ⊢ Iff (LE.le Cardinal.aleph0 (HAdd.hAdd a b)) (Or (LE.le Cardinal.aleph0 a) (L …
  -/
  simp only [← not_lt, add_lt_aleph0_iff, not_and_or]
  /-
    🎉 no goals
  -/


/-- See also `Cardinal.nsmul_lt_aleph0_iff_of_ne_zero` if you already have `n ≠ 0`. -/
theorem nsmul_lt_aleph0_iff {n : ℕ} {a : Cardinal} : n • a < ℵ₀ ↔ n = 0 ∨ a < ℵ₀ := by
  cases n with
  | zero => simpa using nat_lt_aleph0 0
  | succ n =>
      simp only [Nat.succ_ne_zero, false_or]
      induction' n with n ih
      · simp
      rw [succ_nsmul, add_lt_aleph0_iff, ih, and_self_iff]


/-- See also `Cardinal.nsmul_lt_aleph0_iff` for a hypothesis-free version. -/
theorem nsmul_lt_aleph0_iff_of_ne_zero {n : ℕ} {a : Cardinal} (h : n ≠ 0) : n • a < ℵ₀ ↔ a < ℵ₀ :=
  nsmul_lt_aleph0_iff.trans <| or_iff_right h


theorem mul_lt_aleph0 {a b : Cardinal} (ha : a < ℵ₀) (hb : b < ℵ₀) : a * b < ℵ₀ :=
  match a, b, lt_aleph0.1 ha, lt_aleph0.1 hb with
                                   /-
                                     a b : Cardinal.{u_1}
                                     m n : Nat
                                     ha : LT.lt (↑m) Cardinal.aleph0
                                     hb : LT.lt (↑n) Cardinal.aleph0
                                     ⊢ LT.lt (HMul.hMul ↑m ↑n) Cardinal.aleph0
                                   -/
  | _, _, ⟨m, rfl⟩, ⟨n, rfl⟩ => by rw [← Nat.cast_mul]; apply nat_lt_aleph0
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem mul_lt_aleph0_iff {a b : Cardinal} : a * b < ℵ₀ ↔ a = 0 ∨ b = 0 ∨ a < ℵ₀ ∧ b < ℵ₀ := by
  /-
    a b : Cardinal.{u_1}
    ⊢ Iff (LT.lt (HMul.hMul a b) Cardinal.aleph0) (Or (Eq a 0) (Or (Eq b 0) (And ( …
  -/
  refine ⟨fun h => ?_, ?_⟩
    /-
      case refine_1
      a b : Cardinal.{u_1}
      h : LT.lt (HMul.hMul a b) Cardinal.aleph0
      ⊢ Or (Eq a 0) (Or (Eq b 0) (And (LT.lt a Cardinal.aleph0) (LT.lt b Cardinal.al …
    -/
  · by_cases ha : a = 0
      /-
        case pos
        a b : Cardinal.{u_1}
        h : LT.lt (HMul.hMul a b) Cardinal.aleph0
        ha : Eq a 0
        ⊢ Or (Eq a 0) (Or (Eq b 0) (And (LT.lt a Cardinal.aleph0) (LT.lt b Cardinal.al …
      -/
    · exact Or.inl ha
      /-
        🎉 no goals
      -/
    /-
      case neg
      a b : Cardinal.{u_1}
      h : LT.lt (HMul.hMul a b) Cardinal.aleph0
      ha : Not (Eq a 0)
      ⊢ Or (Eq a 0) (Or (Eq b 0) (And (LT.lt a Cardinal.aleph0) (LT.lt b Cardinal.al …
    -/
    right
    /-
      case neg.h
      a b : Cardinal.{u_1}
      h : LT.lt (HMul.hMul a b) Cardinal.aleph0
      ha : Not (Eq a 0)
      ⊢ Or (Eq b 0) (And (LT.lt a Cardinal.aleph0) (LT.lt b Cardinal.aleph0))
    -/
    by_cases hb : b = 0
      /-
        case pos
        a b : Cardinal.{u_1}
        h : LT.lt (HMul.hMul a b) Cardinal.aleph0
        ha : Not (Eq a 0)
        hb : Eq b 0
        ⊢ Or (Eq b 0) (And (LT.lt a Cardinal.aleph0) (LT.lt b Cardinal.aleph0))
      -/
    · exact Or.inl hb
      /-
        🎉 no goals
      -/
    /-
      case neg
      a b : Cardinal.{u_1}
      h : LT.lt (HMul.hMul a b) Cardinal.aleph0
      ha : Not (Eq a 0)
      hb : Not (Eq b 0)
      ⊢ Or (Eq b 0) (And (LT.lt a Cardinal.aleph0) (LT.lt b Cardinal.aleph0))
    -/
    right
    /-
      case neg.h
      a b : Cardinal.{u_1}
      h : LT.lt (HMul.hMul a b) Cardinal.aleph0
      ha : Not (Eq a 0)
      hb : Not (Eq b 0)
      ⊢ And (LT.lt a Cardinal.aleph0) (LT.lt b Cardinal.aleph0)
    -/
    rw [← Ne, ← one_le_iff_ne_zero] at ha hb
    /-
      case neg.h
      a b : Cardinal.{u_1}
      h : LT.lt (HMul.hMul a b) Cardinal.aleph0
      ha : LE.le 1 a
      hb : LE.le 1 b
      ⊢ And (LT.lt a Cardinal.aleph0) (LT.lt b Cardinal.aleph0)
    -/
    constructor
      /-
        case neg.h.left
        a b : Cardinal.{u_1}
        h : LT.lt (HMul.hMul a b) Cardinal.aleph0
        ha : LE.le 1 a
        hb : LE.le 1 b
        ⊢ LT.lt a Cardinal.aleph0
      -/
    · rw [← mul_one a]
      /-
        case neg.h.left
        a b : Cardinal.{u_1}
        h : LT.lt (HMul.hMul a b) Cardinal.aleph0
        ha : LE.le 1 a
        hb : LE.le 1 b
        ⊢ LT.lt (HMul.hMul a 1) Cardinal.aleph0
      -/
      exact (mul_le_mul' le_rfl hb).trans_lt h
      /-
        🎉 no goals
      -/
      /-
        case neg.h.right
        a b : Cardinal.{u_1}
        h : LT.lt (HMul.hMul a b) Cardinal.aleph0
        ha : LE.le 1 a
        hb : LE.le 1 b
        ⊢ LT.lt b Cardinal.aleph0
      -/
    · rw [← one_mul b]
      /-
        case neg.h.right
        a b : Cardinal.{u_1}
        h : LT.lt (HMul.hMul a b) Cardinal.aleph0
        ha : LE.le 1 a
        hb : LE.le 1 b
        ⊢ LT.lt (HMul.hMul 1 b) Cardinal.aleph0
      -/
      exact (mul_le_mul' ha le_rfl).trans_lt h
      /-
        🎉 no goals
      -/
  /-
    case refine_2
    a b : Cardinal.{u_1}
    ⊢ Or (Eq a 0) (Or (Eq b 0) (And (LT.lt a Cardinal.aleph0) (LT.lt b Cardinal.al …
  -/
                                    /-
                                      🎉 no goals
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
  rintro (rfl | rfl | ⟨ha, hb⟩) <;> simp only [*, mul_lt_aleph0, aleph0_pos, zero_mul, mul_zero]
                                    /-
                                      🎉 no goals
                                    -/


/-- See also `Cardinal.aleph0_le_mul_iff`. -/
theorem aleph0_le_mul_iff {a b : Cardinal} : ℵ₀ ≤ a * b ↔ a ≠ 0 ∧ b ≠ 0 ∧ (ℵ₀ ≤ a ∨ ℵ₀ ≤ b) := by
  /-
    a b : Cardinal.{u_1}
    ⊢ Iff (LE.le Cardinal.aleph0 (HMul.hMul a b)) (And (Ne a 0) (And (Ne b 0) (Or  …
  -/
  let h := (@mul_lt_aleph0_iff a b).not
  /-
    a b : Cardinal.{u_1}
    h : Iff (Not (LT.lt (HMul.hMul a b) Cardinal.aleph0)) (Not (Or (Eq a 0) (Or (E …
    ⊢ Iff (LE.le Cardinal.aleph0 (HMul.hMul a b)) (And (Ne a 0) (And (Ne b 0) (Or  …
  -/
  rwa [not_lt, not_or, not_or, not_and_or, not_lt, not_lt] at h
  /-
    🎉 no goals
  -/


/-- See also `Cardinal.aleph0_le_mul_iff'`. -/
theorem aleph0_le_mul_iff' {a b : Cardinal.{u}} : ℵ₀ ≤ a * b ↔ a ≠ 0 ∧ ℵ₀ ≤ b ∨ ℵ₀ ≤ a ∧ b ≠ 0 := by
  /-
    a b : Cardinal.{u}
    ⊢ Iff (LE.le Cardinal.aleph0 (HMul.hMul a b)) (Or (And (Ne a 0) (LE.le Cardina …
  -/
  have : ∀ {a : Cardinal.{u}}, ℵ₀ ≤ a → a ≠ 0 := fun a => ne_bot_of_le_ne_bot aleph0_ne_zero a
  /-
    a b : Cardinal.{u}
    this : ∀ {a : Cardinal.{u}}, LE.le Cardinal.aleph0 a → Ne a 0
    ⊢ Iff (LE.le Cardinal.aleph0 (HMul.hMul a b)) (Or (And (Ne a 0) (LE.le Cardina …
  -/
  simp only [aleph0_le_mul_iff, and_or_left, and_iff_right_of_imp this, @and_left_comm (a ≠ 0)]
  /-
    a b : Cardinal.{u}
    this : ∀ {a : Cardinal.{u}}, LE.le Cardinal.aleph0 a → Ne a 0
    ⊢ Iff (Or (And (Ne b 0) (LE.le Cardinal.aleph0 a)) (And (Ne a 0) (LE.le Cardin …
  -/
  simp only [and_comm, or_comm]
  /-
    🎉 no goals
  -/


theorem mul_lt_aleph0_iff_of_ne_zero {a b : Cardinal} (ha : a ≠ 0) (hb : b ≠ 0) :
                                       /-
                                         a b : Cardinal.{u_1}
                                         ha : Ne a 0
                                         hb : Ne b 0
                                         ⊢ Iff (LT.lt (HMul.hMul a b) Cardinal.aleph0) (And (LT.lt a Cardinal.aleph0) ( …
                                       -/
    a * b < ℵ₀ ↔ a < ℵ₀ ∧ b < ℵ₀ := by simp [mul_lt_aleph0_iff, ha, hb]
                                       /-
                                         🎉 no goals
                                       -/


theorem power_lt_aleph0 {a b : Cardinal} (ha : a < ℵ₀) (hb : b < ℵ₀) : a ^ b < ℵ₀ :=
  match a, b, lt_aleph0.1 ha, lt_aleph0.1 hb with
                                   /-
                                     a b : Cardinal.{u_1}
                                     m n : Nat
                                     ha : LT.lt (↑m) Cardinal.aleph0
                                     hb : LT.lt (↑n) Cardinal.aleph0
                                     ⊢ LT.lt (HPow.hPow ↑m ↑n) Cardinal.aleph0
                                   -/
  | _, _, ⟨m, rfl⟩, ⟨n, rfl⟩ => by rw [power_natCast, ← Nat.cast_pow]; apply nat_lt_aleph0
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem eq_one_iff_unique {α : Type*} : #α = 1 ↔ Subsingleton α ∧ Nonempty α :=
  calc
    #α = 1 ↔ #α ≤ 1 ∧ 1 ≤ #α := le_antisymm_iff
    _ ↔ Subsingleton α ∧ Nonempty α :=
      le_one_iff_subsingleton.and (one_le_iff_ne_zero.trans mk_ne_zero_iff)


theorem infinite_iff {α : Type u} : Infinite α ↔ ℵ₀ ≤ #α := by
  /-
    α : Type u
    ⊢ Iff (Infinite α) (LE.le Cardinal.aleph0 (Cardinal.mk α))
  -/
  rw [← not_lt, lt_aleph0_iff_finite, not_finite_iff_infinite]
  /-
    🎉 no goals
  -/


lemma aleph0_le_mk_iff : ℵ₀ ≤ #α ↔ Infinite α := infinite_iff.symm

                                                  /-
                                                    α : Type u
                                                    ⊢ Iff (LT.lt (Cardinal.mk α) Cardinal.aleph0) (Finite α)
                                                  -/
lemma mk_lt_aleph0_iff : #α < ℵ₀ ↔ Finite α := by simp [← not_le, aleph0_le_mk_iff]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem aleph0_le_mk (α : Type u) [Infinite α] : ℵ₀ ≤ #α :=
  infinite_iff.1 ‹_›


@[simp]
theorem mk_eq_aleph0 (α : Type*) [Countable α] [Infinite α] : #α = ℵ₀ :=
  mk_le_aleph0.antisymm <| aleph0_le_mk _


theorem denumerable_iff {α : Type u} : Nonempty (Denumerable α) ↔ #α = ℵ₀ :=
  ⟨fun ⟨h⟩ => mk_congr ((@Denumerable.eqv α h).trans Equiv.ulift.symm), fun h => by
    /-
      α : Type u
      h : Eq (Cardinal.mk α) Cardinal.aleph0
      ⊢ Nonempty (Denumerable α)
    -/
    cases' Quotient.exact h with f
    /-
      case intro
      α : Type u
      h : Eq (Cardinal.mk α) Cardinal.aleph0
      f : Equiv α (ULift.{u, 0} Nat)
      ⊢ Nonempty (Denumerable α)
    -/
    exact ⟨Denumerable.mk' <| f.trans Equiv.ulift⟩⟩
    /-
      🎉 no goals
    -/


theorem mk_denumerable (α : Type u) [Denumerable α] : #α = ℵ₀ :=
  denumerable_iff.1 ⟨‹_›⟩


theorem _root_.Set.countable_infinite_iff_nonempty_denumerable {α : Type*} {s : Set α} :
    s.Countable ∧ s.Infinite ↔ Nonempty (Denumerable s) := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Iff (And s.Countable s.Infinite) (Nonempty (Denumerable ↑s))
  -/
  rw [nonempty_denumerable_iff, ← Set.infinite_coe_iff, countable_coe_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem aleph0_add_aleph0 : ℵ₀ + ℵ₀ = ℵ₀ :=
  mk_denumerable _


theorem aleph0_mul_aleph0 : ℵ₀ * ℵ₀ = ℵ₀ :=
  mk_denumerable _


@[simp]
theorem nat_mul_aleph0 {n : ℕ} (hn : n ≠ 0) : ↑n * ℵ₀ = ℵ₀ :=
  le_antisymm (lift_mk_fin n ▸ mk_le_aleph0) <|
    le_mul_of_one_le_left (zero_le _) <| by
      /-
        n : Nat
        hn : Ne n 0
        ⊢ LE.le 1 ↑n
      -/
      rwa [← Nat.cast_one, Nat.cast_le, Nat.one_le_iff_ne_zero]
      /-
        🎉 no goals
      -/


@[simp]
                                                                /-
                                                                  n : Nat
                                                                  hn : Ne n 0
                                                                  ⊢ Eq (HMul.hMul Cardinal.aleph0 ↑n) Cardinal.aleph0
                                                                -/
theorem aleph0_mul_nat {n : ℕ} (hn : n ≠ 0) : ℵ₀ * n = ℵ₀ := by rw [mul_comm, nat_mul_aleph0 hn]
                                                                /-
                                                                  🎉 no goals
                                                                -/

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem ofNat_mul_aleph0 {n : ℕ} [Nat.AtLeastTwo n] : no_index (OfNat.ofNat n) * ℵ₀ = ℵ₀ :=
  nat_mul_aleph0 (NeZero.ne n)

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem aleph0_mul_ofNat {n : ℕ} [Nat.AtLeastTwo n] : ℵ₀ * no_index (OfNat.ofNat n) = ℵ₀ :=
  aleph0_mul_nat (NeZero.ne n)


@[simp]
theorem add_le_aleph0 {c₁ c₂ : Cardinal} : c₁ + c₂ ≤ ℵ₀ ↔ c₁ ≤ ℵ₀ ∧ c₂ ≤ ℵ₀ :=
  ⟨fun h => ⟨le_self_add.trans h, le_add_self.trans h⟩, fun h =>
    aleph0_add_aleph0 ▸ add_le_add h.1 h.2⟩


@[simp]
theorem aleph0_add_nat (n : ℕ) : ℵ₀ + n = ℵ₀ :=
  (add_le_aleph0.2 ⟨le_rfl, (nat_lt_aleph0 n).le⟩).antisymm le_self_add


@[simp]
                                                    /-
                                                      n : Nat
                                                      ⊢ Eq (HAdd.hAdd (↑n) Cardinal.aleph0) Cardinal.aleph0
                                                    -/
theorem nat_add_aleph0 (n : ℕ) : ↑n + ℵ₀ = ℵ₀ := by rw [add_comm, aleph0_add_nat]
                                                    /-
                                                      🎉 no goals
                                                    -/

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem ofNat_add_aleph0 {n : ℕ} [Nat.AtLeastTwo n] : no_index (OfNat.ofNat n) + ℵ₀ = ℵ₀ :=
  nat_add_aleph0 n

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem aleph0_add_ofNat {n : ℕ} [Nat.AtLeastTwo n] : ℵ₀ + no_index (OfNat.ofNat n) = ℵ₀ :=
  aleph0_add_nat n


theorem exists_nat_eq_of_le_nat {c : Cardinal} {n : ℕ} (h : c ≤ n) : ∃ m, m ≤ n ∧ c = m := by
  /-
    c : Cardinal.{u_1}
    n : Nat
    h : LE.le c ↑n
    ⊢ Exists fun m => And (LE.le m n) (Eq c ↑m)
  -/
  lift c to ℕ using h.trans_lt (nat_lt_aleph0 _)
  /-
    case intro
    n c : Nat
    h : LE.le ↑c ↑n
    ⊢ Exists fun m => And (LE.le m n) (Eq ↑c ↑m)
  -/
  exact ⟨c, mod_cast h, rfl⟩
  /-
    🎉 no goals
  -/


theorem mk_int : #ℤ = ℵ₀ :=
  mk_denumerable ℤ


theorem mk_pNat : #ℕ+ = ℵ₀ :=
  mk_denumerable ℕ+


theorem mk_empty : #Empty = 0 :=
  mk_eq_zero _


theorem mk_pempty : #PEmpty = 0 :=
  mk_eq_zero _


theorem mk_punit : #PUnit = 1 :=
  mk_eq_one PUnit


theorem mk_unit : #Unit = 1 :=
  mk_punit


@[simp] theorem mk_additive : #(Additive α) = #α := rfl


@[simp] theorem mk_multiplicative : #(Multiplicative α) = #α := rfl


@[to_additive (attr := simp)] theorem mk_mulOpposite : #(MulOpposite α) = #α :=
  mk_congr MulOpposite.opEquiv.symm


theorem mk_singleton {α : Type u} (x : α) : #({x} : Set α) = 1 :=
  mk_eq_one _


theorem mk_plift_true : #(PLift True) = 1 :=
  mk_eq_one _


theorem mk_plift_false : #(PLift False) = 0 :=
  mk_eq_zero _


@[simp]
theorem mk_vector (α : Type u) (n : ℕ) : #(List.Vector α n) = #α ^ n :=
                                                    /-
                                                      α : Type u
                                                      n : Nat
                                                      ⊢ Eq (Cardinal.mk (Fin n → α)) (HPow.hPow (Cardinal.mk α) n)
                                                    -/
  (mk_congr (Equiv.vectorEquivFin α n)).trans <| by simp
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem mk_list_eq_sum_pow (α : Type u) : #(List α) = sum fun n : ℕ => #α ^ n :=
  calc
    #(List α) = #(Σn, List.Vector α n) := mk_congr (Equiv.sigmaFiberEquiv List.length).symm
                                      /-
                                        α : Type u
                                        ⊢ Eq (Cardinal.mk (Sigma fun n => List.Vector α n)) (Cardinal.sum fun n => HPo …
                                      -/
    _ = sum fun n : ℕ => #α ^ n := by simp
                                      /-
                                        🎉 no goals
                                      -/


theorem mk_quot_le {α : Type u} {r : α → α → Prop} : #(Quot r) ≤ #α :=
  mk_le_of_surjective Quot.exists_rep


theorem mk_quotient_le {α : Type u} {s : Setoid α} : #(Quotient s) ≤ #α :=
  mk_quot_le


theorem mk_subtype_le_of_subset {α : Type u} {p q : α → Prop} (h : ∀ ⦃x⦄, p x → q x) :
    #(Subtype p) ≤ #(Subtype q) :=
  ⟨Embedding.subtypeMap (Embedding.refl α) h⟩


theorem mk_emptyCollection (α : Type u) : #(∅ : Set α) = 0 :=
  mk_eq_zero _


theorem mk_emptyCollection_iff {α : Type u} {s : Set α} : #s = 0 ↔ s = ∅ := by
  /-
    α : Type u
    s : Set α
    ⊢ Iff (Eq (Cardinal.mk ↑s) 0) (Eq s EmptyCollection.emptyCollection)
  -/
  constructor
    /-
      case mp
      α : Type u
      s : Set α
      ⊢ Eq (Cardinal.mk ↑s) 0 → Eq s EmptyCollection.emptyCollection
    -/
  · intro h
    /-
      case mp
      α : Type u
      s : Set α
      h : Eq (Cardinal.mk ↑s) 0
      ⊢ Eq s EmptyCollection.emptyCollection
    -/
    rw [mk_eq_zero_iff] at h
    /-
      case mp
      α : Type u
      s : Set α
      h : IsEmpty ↑s
      ⊢ Eq s EmptyCollection.emptyCollection
    -/
    exact eq_empty_iff_forall_not_mem.2 fun x hx => h.elim' ⟨x, hx⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      s : Set α
      ⊢ Eq s EmptyCollection.emptyCollection → Eq (Cardinal.mk ↑s) 0
    -/
  · rintro rfl
    /-
      case mpr
      α : Type u
      ⊢ Eq (Cardinal.mk ↑EmptyCollection.emptyCollection) 0
    -/
    exact mk_emptyCollection _
    /-
      🎉 no goals
    -/


@[simp]
theorem mk_univ {α : Type u} : #(@univ α) = #α :=
  mk_congr (Equiv.Set.univ α)


@[simp] lemma mk_setProd {α β : Type u} (s : Set α) (t : Set β) : #(s ×ˢ t) = #s * #t := by
  /-
    α β : Type u
    s : Set α
    t : Set β
    ⊢ Eq (Cardinal.mk ↑(SProd.sprod s t)) (HMul.hMul (Cardinal.mk ↑s) (Cardinal.mk …
  -/
  rw [mul_def, mk_congr (Equiv.Set.prod ..)]
  /-
    🎉 no goals
  -/


theorem mk_image_le {α β : Type u} {f : α → β} {s : Set α} : #(f '' s) ≤ #s :=
  mk_le_of_surjective surjective_onto_image


lemma mk_image2_le {α β γ : Type u} {f : α → β → γ} {s : Set α} {t : Set β} :
    #(image2 f s t) ≤ #s * #t := by
  /-
    α β γ : Type u
    f : α → β → γ
    s : Set α
    t : Set β
    ⊢ LE.le (Cardinal.mk ↑(Set.image2 f s t)) (HMul.hMul (Cardinal.mk ↑s) (Cardina …
  -/
  rw [← image_uncurry_prod, ← mk_setProd]
  /-
    α β γ : Type u
    f : α → β → γ
    s : Set α
    t : Set β
    ⊢ LE.le (Cardinal.mk ↑(Set.image (Function.uncurry f) (SProd.sprod s t))) (Car …
  -/
  exact mk_image_le
  /-
    🎉 no goals
  -/


theorem mk_image_le_lift {α : Type u} {β : Type v} {f : α → β} {s : Set α} :
    lift.{u} #(f '' s) ≤ lift.{v} #s :=
  lift_mk_le.{0}.mpr ⟨Embedding.ofSurjective _ surjective_onto_image⟩


theorem mk_range_le {α β : Type u} {f : α → β} : #(range f) ≤ #α :=
  mk_le_of_surjective surjective_onto_range


theorem mk_range_le_lift {α : Type u} {β : Type v} {f : α → β} :
    lift.{u} #(range f) ≤ lift.{v} #α :=
  lift_mk_le.{0}.mpr ⟨Embedding.ofSurjective _ surjective_onto_range⟩


theorem mk_range_eq (f : α → β) (h : Injective f) : #(range f) = #α :=
  mk_congr (Equiv.ofInjective f h).symm


theorem mk_range_eq_lift {α : Type u} {β : Type v} {f : α → β} (hf : Injective f) :
    lift.{max u w} #(range f) = lift.{max v w} #α :=
  lift_mk_eq.{v,u,w}.mpr ⟨(Equiv.ofInjective f hf).symm⟩


theorem mk_range_eq_of_injective {α : Type u} {β : Type v} {f : α → β} (hf : Injective f) :
    lift.{u} #(range f) = lift.{v} #α :=
  lift_mk_eq'.mpr ⟨(Equiv.ofInjective f hf).symm⟩


lemma lift_mk_le_lift_mk_of_injective {α : Type u} {β : Type v} {f : α → β} (hf : Injective f) :
    Cardinal.lift.{v} (#α) ≤ Cardinal.lift.{u} (#β) := by
  /-
    α : Type u
    β : Type v
    f : α → β
    hf : Function.Injective f
    ⊢ LE.le (Cardinal.lift.{v, u} (Cardinal.mk α)) (Cardinal.lift.{u, v} (Cardinal …
  -/
  rw [← Cardinal.mk_range_eq_of_injective hf]
  /-
    α : Type u
    β : Type v
    f : α → β
    hf : Function.Injective f
    ⊢ LE.le (Cardinal.lift.{u, v} (Cardinal.mk ↑(Set.range f))) (Cardinal.lift.{u, …
  -/
  exact Cardinal.lift_le.2 (Cardinal.mk_set_le _)
  /-
    🎉 no goals
  -/


lemma lift_mk_le_lift_mk_of_surjective {α : Type u} {β : Type v} {f : α → β} (hf : Surjective f) :
    Cardinal.lift.{u} (#β) ≤ Cardinal.lift.{v} (#α) :=
  lift_mk_le_lift_mk_of_injective (injective_surjInv hf)


theorem mk_image_eq_of_injOn {α β : Type u} (f : α → β) (s : Set α) (h : InjOn f s) :
    #(f '' s) = #s :=
  mk_congr (Equiv.Set.imageOfInjOn f s h).symm


theorem mk_image_eq_of_injOn_lift {α : Type u} {β : Type v} (f : α → β) (s : Set α)
    (h : InjOn f s) : lift.{u} #(f '' s) = lift.{v} #s :=
  lift_mk_eq.{v, u, 0}.mpr ⟨(Equiv.Set.imageOfInjOn f s h).symm⟩


theorem mk_image_eq {α β : Type u} {f : α → β} {s : Set α} (hf : Injective f) : #(f '' s) = #s :=
  mk_image_eq_of_injOn _ _ hf.injOn


theorem mk_image_eq_lift {α : Type u} {β : Type v} (f : α → β) (s : Set α) (h : Injective f) :
    lift.{u} #(f '' s) = lift.{v} #s :=
  mk_image_eq_of_injOn_lift _ _ h.injOn


theorem mk_iUnion_le_sum_mk {α ι : Type u} {f : ι → Set α} : #(⋃ i, f i) ≤ sum fun i => #(f i) :=
  calc
    #(⋃ i, f i) ≤ #(Σi, f i) := mk_le_of_surjective (Set.sigmaToiUnion_surjective f)
    _ = sum fun i => #(f i) := mk_sigma _


theorem mk_iUnion_le_sum_mk_lift {α : Type u} {ι : Type v} {f : ι → Set α} :
    lift.{v} #(⋃ i, f i) ≤ sum fun i => #(f i) :=
  calc
    lift.{v} #(⋃ i, f i) ≤ #(Σi, f i) :=
      mk_le_of_surjective <| ULift.up_surjective.comp (Set.sigmaToiUnion_surjective f)
    _ = sum fun i => #(f i) := mk_sigma _


theorem mk_iUnion_eq_sum_mk {α ι : Type u} {f : ι → Set α}
    (h : Pairwise (Disjoint on f)) : #(⋃ i, f i) = sum fun i => #(f i) :=
  calc
    #(⋃ i, f i) = #(Σi, f i) := mk_congr (Set.unionEqSigmaOfDisjoint h)
    _ = sum fun i => #(f i) := mk_sigma _


theorem mk_iUnion_eq_sum_mk_lift {α : Type u} {ι : Type v} {f : ι → Set α}
    (h : Pairwise (Disjoint on f)) :
    lift.{v} #(⋃ i, f i) = sum fun i => #(f i) :=
  calc
    lift.{v} #(⋃ i, f i) = #(Σi, f i) :=
      mk_congr <| .trans Equiv.ulift (Set.unionEqSigmaOfDisjoint h)
    _ = sum fun i => #(f i) := mk_sigma _


theorem mk_iUnion_le {α ι : Type u} (f : ι → Set α) : #(⋃ i, f i) ≤ #ι * ⨆ i, #(f i) :=
  mk_iUnion_le_sum_mk.trans (sum_le_iSup _)


theorem mk_iUnion_le_lift {α : Type u} {ι : Type v} (f : ι → Set α) :
    lift.{v} #(⋃ i, f i) ≤ lift.{u} #ι * ⨆ i, lift.{v} #(f i) := by
  /-
    α : Type u
    ι : Type v
    f : ι → Set α
    ⊢ LE.le (Cardinal.lift.{v, u} (Cardinal.mk ↑(Set.iUnion fun i => f i))) (HMul. …
  -/
  refine mk_iUnion_le_sum_mk_lift.trans <| Eq.trans_le ?_ (sum_le_iSup_lift _)
  /-
    α : Type u
    ι : Type v
    f : ι → Set α
    ⊢ Eq (Cardinal.sum fun i => Cardinal.mk ↑(f i)) (Cardinal.sum fun i => Cardina …
  -/
  rw [← lift_sum, lift_id'.{_,u}]
  /-
    🎉 no goals
  -/


theorem mk_sUnion_le {α : Type u} (A : Set (Set α)) : #(⋃₀ A) ≤ #A * ⨆ s : A, #s := by
  /-
    α : Type u
    A : Set (Set α)
    ⊢ LE.le (Cardinal.mk ↑A.sUnion) (HMul.hMul (Cardinal.mk ↑A) (iSup fun s => Car …
  -/
  rw [sUnion_eq_iUnion]
  /-
    α : Type u
    A : Set (Set α)
    ⊢ LE.le (Cardinal.mk ↑(Set.iUnion fun i => ↑i)) (HMul.hMul (Cardinal.mk ↑A) (i …
  -/
  apply mk_iUnion_le
  /-
    🎉 no goals
  -/


theorem mk_biUnion_le {ι α : Type u} (A : ι → Set α) (s : Set ι) :
    #(⋃ x ∈ s, A x) ≤ #s * ⨆ x : s, #(A x.1) := by
  /-
    ι α : Type u
    A : ι → Set α
    s : Set ι
    ⊢ LE.le (Cardinal.mk ↑(Set.iUnion fun x => Set.iUnion fun h => A x)) (HMul.hMu …
  -/
  rw [biUnion_eq_iUnion]
  /-
    ι α : Type u
    A : ι → Set α
    s : Set ι
    ⊢ LE.le (Cardinal.mk ↑(Set.iUnion fun x => A ↑x)) (HMul.hMul (Cardinal.mk ↑s)  …
  -/
  apply mk_iUnion_le
  /-
    🎉 no goals
  -/


theorem mk_biUnion_le_lift {α : Type u} {ι : Type v} (A : ι → Set α) (s : Set ι) :
    lift.{v} #(⋃ x ∈ s, A x) ≤ lift.{u} #s * ⨆ x : s, lift.{v} #(A x.1) := by
  /-
    α : Type u
    ι : Type v
    A : ι → Set α
    s : Set ι
    ⊢ LE.le (Cardinal.lift.{v, u} (Cardinal.mk ↑(Set.iUnion fun x => Set.iUnion fu …
  -/
  rw [biUnion_eq_iUnion]
  /-
    α : Type u
    ι : Type v
    A : ι → Set α
    s : Set ι
    ⊢ LE.le (Cardinal.lift.{v, u} (Cardinal.mk ↑(Set.iUnion fun x => A ↑x))) (HMul …
  -/
  apply mk_iUnion_le_lift
  /-
    🎉 no goals
  -/


theorem finset_card_lt_aleph0 (s : Finset α) : #(↑s : Set α) < ℵ₀ :=
  lt_aleph0_of_finite _


theorem mk_set_eq_nat_iff_finset {α} {s : Set α} {n : ℕ} :
    #s = n ↔ ∃ t : Finset α, (t : Set α) = s ∧ t.card = n := by
  /-
    α : Type u_1
    s : Set α
    n : Nat
    ⊢ Iff (Eq (Cardinal.mk ↑s) ↑n) (Exists fun t => And (Eq (↑t) s) (Eq t.card n))
  -/
  constructor
    /-
      case mp
      α : Type u_1
      s : Set α
      n : Nat
      ⊢ Eq (Cardinal.mk ↑s) ↑n → Exists fun t => And (Eq (↑t) s) (Eq t.card n)
    -/
  · intro h
    /-
      case mp
      α : Type u_1
      s : Set α
      n : Nat
      h : Eq (Cardinal.mk ↑s) ↑n
      ⊢ Exists fun t => And (Eq (↑t) s) (Eq t.card n)
    -/
    lift s to Finset α using lt_aleph0_iff_set_finite.1 (h.symm ▸ nat_lt_aleph0 n)
    /-
      case mp.intro
      α : Type u_1
      n : Nat
      s : Finset α
      h : Eq (Cardinal.mk ↑↑s) ↑n
      ⊢ Exists fun t => And (Eq ↑t ↑s) (Eq t.card n)
    -/
    simpa using h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      s : Set α
      n : Nat
      ⊢ (Exists fun t => And (Eq (↑t) s) (Eq t.card n)) → Eq (Cardinal.mk ↑s) ↑n
    -/
  · rintro ⟨t, rfl, rfl⟩
    /-
      case mpr.intro.intro
      α : Type u_1
      t : Finset α
      ⊢ Eq (Cardinal.mk ↑↑t) ↑t.card
    -/
    exact mk_coe_finset
    /-
      🎉 no goals
    -/


theorem mk_eq_nat_iff_finset {n : ℕ} :
    #α = n ↔ ∃ t : Finset α, (t : Set α) = univ ∧ t.card = n := by
  /-
    α : Type u
    n : Nat
    ⊢ Iff (Eq (Cardinal.mk α) ↑n) (Exists fun t => And (Eq (↑t) Set.univ) (Eq t.ca …
  -/
  rw [← mk_univ, mk_set_eq_nat_iff_finset]
  /-
    🎉 no goals
  -/


theorem mk_eq_nat_iff_fintype {n : ℕ} : #α = n ↔ ∃ h : Fintype α, @Fintype.card α h = n := by
  /-
    α : Type u
    n : Nat
    ⊢ Iff (Eq (Cardinal.mk α) ↑n) (Exists fun h => Eq (Fintype.card α) n)
  -/
  rw [mk_eq_nat_iff_finset]
  /-
    α : Type u
    n : Nat
    ⊢ Iff (Exists fun t => And (Eq (↑t) Set.univ) (Eq t.card n)) (Exists fun h =>  …
  -/
  constructor
    /-
      case mp
      α : Type u
      n : Nat
      ⊢ (Exists fun t => And (Eq (↑t) Set.univ) (Eq t.card n)) → Exists fun h => Eq  …
    -/
  · rintro ⟨t, ht, hn⟩
    /-
      case mp.intro.intro
      α : Type u
      n : Nat
      t : Finset α
      ht : Eq (↑t) Set.univ
      hn : Eq t.card n
      ⊢ Exists fun h => Eq (Fintype.card α) n
    -/
    exact ⟨⟨t, eq_univ_iff_forall.1 ht⟩, hn⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      n : Nat
      ⊢ (Exists fun h => Eq (Fintype.card α) n) → Exists fun t => And (Eq (↑t) Set.u …
    -/
  · rintro ⟨⟨t, ht⟩, hn⟩
    /-
      case mpr.intro.mk
      α : Type u
      n : Nat
      t : Finset α
      ht : ∀ (x : α), Membership.mem t x
      hn : Eq (Fintype.card α) n
      ⊢ Exists fun t => And (Eq (↑t) Set.univ) (Eq t.card n)
    -/
    exact ⟨t, eq_univ_iff_forall.2 ht, hn⟩
    /-
      🎉 no goals
    -/


theorem mk_union_add_mk_inter {α : Type u} {S T : Set α} :
    #(S ∪ T : Set α) + #(S ∩ T : Set α) = #S + #T := by
  classical
  exact Quot.sound ⟨Equiv.Set.unionSumInter S T⟩


/-- The cardinality of a union is at most the sum of the cardinalities
of the two sets. -/
theorem mk_union_le {α : Type u} (S T : Set α) : #(S ∪ T : Set α) ≤ #S + #T :=
  @mk_union_add_mk_inter α S T ▸ self_le_add_right #(S ∪ T : Set α) #(S ∩ T : Set α)


theorem mk_union_of_disjoint {α : Type u} {S T : Set α} (H : Disjoint S T) :
    #(S ∪ T : Set α) = #S + #T := by
  classical
  exact Quot.sound ⟨Equiv.Set.union H⟩


theorem mk_insert {α : Type u} {s : Set α} {a : α} (h : a ∉ s) :
    #(insert a s : Set α) = #s + 1 := by
  /-
    α : Type u
    s : Set α
    a : α
    h : Not (Membership.mem s a)
    ⊢ Eq (Cardinal.mk ↑(Insert.insert a s)) (HAdd.hAdd (Cardinal.mk ↑s) 1)
  -/
  rw [← union_singleton, mk_union_of_disjoint, mk_singleton]
  /-
    α : Type u
    s : Set α
    a : α
    h : Not (Membership.mem s a)
    ⊢ Disjoint s (Singleton.singleton a)
  -/
  simpa
  /-
    🎉 no goals
  -/


theorem mk_insert_le {α : Type u} {s : Set α} {a : α} : #(insert a s : Set α) ≤ #s + 1 := by
  /-
    α : Type u
    s : Set α
    a : α
    ⊢ LE.le (Cardinal.mk ↑(Insert.insert a s)) (HAdd.hAdd (Cardinal.mk ↑s) 1)
  -/
  by_cases h : a ∈ s
    /-
      case pos
      α : Type u
      s : Set α
      a : α
      h : Membership.mem s a
      ⊢ LE.le (Cardinal.mk ↑(Insert.insert a s)) (HAdd.hAdd (Cardinal.mk ↑s) 1)
    -/
  · simp only [insert_eq_of_mem h, self_le_add_right]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u
      s : Set α
      a : α
      h : Not (Membership.mem s a)
      ⊢ LE.le (Cardinal.mk ↑(Insert.insert a s)) (HAdd.hAdd (Cardinal.mk ↑s) 1)
    -/
  · rw [mk_insert h]
    /-
      🎉 no goals
    -/


theorem mk_sum_compl {α} (s : Set α) : #s + #(sᶜ : Set α) = #α := by
  classical
  exact mk_congr (Equiv.Set.sumCompl s)


theorem mk_le_mk_of_subset {α} {s t : Set α} (h : s ⊆ t) : #s ≤ #t :=
  ⟨Set.embeddingOfSubset s t h⟩


theorem mk_le_iff_forall_finset_subset_card_le {α : Type u} {n : ℕ} {t : Set α} :
    #t ≤ n ↔ ∀ s : Finset α, (s : Set α) ⊆ t → s.card ≤ n := by
  /-
    α : Type u
    n : Nat
    t : Set α
    ⊢ Iff (LE.le (Cardinal.mk ↑t) ↑n) (∀ (s : Finset α), HasSubset.Subset (↑s) t → …
  -/
  refine ⟨fun H s hs ↦ by simpa using (mk_le_mk_of_subset hs).trans H, fun H ↦ ?_⟩
  /-
    α : Type u
    n : Nat
    t : Set α
    H : ∀ (s : Finset α), HasSubset.Subset (↑s) t → LE.le s.card n
    ⊢ LE.le (Cardinal.mk ↑t) ↑n
  -/
  apply card_le_of (fun s ↦ ?_)
  classical
  let u : Finset α := s.image Subtype.val
  have : u.card = s.card := Finset.card_image_of_injOn Subtype.coe_injective.injOn
  rw [← this]
  apply H
  simp only [u, Finset.coe_image, image_subset_iff, Subtype.coe_preimage_self, subset_univ]


theorem mk_subtype_mono {p q : α → Prop} (h : ∀ x, p x → q x) :
    #{ x // p x } ≤ #{ x // q x } :=
  ⟨embeddingOfSubset _ _ h⟩


theorem le_mk_diff_add_mk (S T : Set α) : #S ≤ #(S \ T : Set α) + #T :=
  (mk_le_mk_of_subset <| subset_diff_union _ _).trans <| mk_union_le _ _


theorem mk_diff_add_mk {S T : Set α} (h : T ⊆ S) : #(S \ T : Set α) + #T = #S := by
  /-
    α : Type u
    S T : Set α
    h : HasSubset.Subset T S
    ⊢ Eq (HAdd.hAdd (Cardinal.mk ↑(SDiff.sdiff S T)) (Cardinal.mk ↑T)) (Cardinal.m …
  -/
  refine (mk_union_of_disjoint <| ?_).symm.trans <| by rw [diff_union_of_subset h]
  /-
    α : Type u
    S T : Set α
    h : HasSubset.Subset T S
    ⊢ Disjoint (SDiff.sdiff S T) T
  -/
  exact disjoint_sdiff_self_left
  /-
    🎉 no goals
  -/


theorem mk_union_le_aleph0 {α} {P Q : Set α} :
    #(P ∪ Q : Set α) ≤ ℵ₀ ↔ #P ≤ ℵ₀ ∧ #Q ≤ ℵ₀ := by
  simp only [le_aleph0_iff_subtype_countable, mem_union, setOf_mem_eq, Set.union_def,
    ← countable_union]


theorem mk_subtype_of_equiv {α β : Type u} (p : β → Prop) (e : α ≃ β) :
    #{ a : α // p (e a) } = #{ b : β // p b } :=
  mk_congr (Equiv.subtypeEquivOfSubtype e)


theorem mk_sep (s : Set α) (t : α → Prop) : #({ x ∈ s | t x } : Set α) = #{ x : s | t x.1 } :=
  mk_congr (Equiv.Set.sep s t)


theorem mk_preimage_of_injective_lift {α : Type u} {β : Type v} (f : α → β) (s : Set β)
    (h : Injective f) : lift.{v} #(f ⁻¹' s) ≤ lift.{u} #s := by
  /-
    α : Type u
    β : Type v
    f : α → β
    s : Set β
    h : Function.Injective f
    ⊢ LE.le (Cardinal.lift.{v, u} (Cardinal.mk ↑(Set.preimage f s))) (Cardinal.lif …
  -/
  rw [lift_mk_le.{0}]
  -- Porting note: Needed to insert `mem_preimage.mp` below
  /-
    α : Type u
    β : Type v
    f : α → β
    s : Set β
    h : Function.Injective f
    ⊢ Nonempty (Function.Embedding ↑(Set.preimage f s) ↑s)
  -/
  use Subtype.coind (fun x => f x.1) fun x => mem_preimage.mp x.2
  /-
    case inj'
    α : Type u
    β : Type v
    f : α → β
    s : Set β
    h : Function.Injective f
    ⊢ Function.Injective (Subtype.coind (fun x => f ↑x) ⋯)
  -/
  apply Subtype.coind_injective; exact h.comp Subtype.val_injective
                                 /-
                                   🎉 no goals
                                 -/


theorem mk_preimage_of_subset_range_lift {α : Type u} {β : Type v} (f : α → β) (s : Set β)
    (h : s ⊆ range f) : lift.{u} #s ≤ lift.{v} #(f ⁻¹' s) := by
  /-
    α : Type u
    β : Type v
    f : α → β
    s : Set β
    h : HasSubset.Subset s (Set.range f)
    ⊢ LE.le (Cardinal.lift.{u, v} (Cardinal.mk ↑s)) (Cardinal.lift.{v, u} (Cardina …
  -/
  rw [← image_preimage_eq_iff] at h
  /-
    α : Type u
    β : Type v
    f : α → β
    s : Set β
    h : Eq (Set.image f (Set.preimage f s)) s
    ⊢ LE.le (Cardinal.lift.{u, v} (Cardinal.mk ↑s)) (Cardinal.lift.{v, u} (Cardina …
  -/
  nth_rewrite 1 [← h]
  /-
    α : Type u
    β : Type v
    f : α → β
    s : Set β
    h : Eq (Set.image f (Set.preimage f s)) s
    ⊢ LE.le (Cardinal.lift.{u, v} (Cardinal.mk ↑(Set.image f (Set.preimage f s)))) …
  -/
  apply mk_image_le_lift
  /-
    🎉 no goals
  -/


theorem mk_preimage_of_injective_of_subset_range_lift {β : Type v} (f : α → β) (s : Set β)
    (h : Injective f) (h2 : s ⊆ range f) : lift.{v} #(f ⁻¹' s) = lift.{u} #s :=
  le_antisymm (mk_preimage_of_injective_lift f s h) (mk_preimage_of_subset_range_lift f s h2)


theorem mk_preimage_of_injective_of_subset_range (f : α → β) (s : Set β) (h : Injective f)
    (h2 : s ⊆ range f) : #(f ⁻¹' s) = #s := by
  /-
    α β : Type u
    f : α → β
    s : Set β
    h : Function.Injective f
    h2 : HasSubset.Subset s (Set.range f)
    ⊢ Eq (Cardinal.mk ↑(Set.preimage f s)) (Cardinal.mk ↑s)
  -/
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
  convert mk_preimage_of_injective_of_subset_range_lift.{u, u} f s h h2 using 1 <;> rw [lift_id]
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


theorem mk_preimage_of_injective (f : α → β) (s : Set β) (h : Injective f) :
    #(f ⁻¹' s) ≤ #s := by
  /-
    α β : Type u
    f : α → β
    s : Set β
    h : Function.Injective f
    ⊢ LE.le (Cardinal.mk ↑(Set.preimage f s)) (Cardinal.mk ↑s)
  -/
  rw [← lift_id #(↑(f ⁻¹' s)), ← lift_id #(↑s)]
  /-
    α β : Type u
    f : α → β
    s : Set β
    h : Function.Injective f
    ⊢ LE.le (Cardinal.lift.{u, u} (Cardinal.mk ↑(Set.preimage f s))) (Cardinal.lif …
  -/
  exact mk_preimage_of_injective_lift f s h
  /-
    🎉 no goals
  -/


theorem mk_preimage_of_subset_range (f : α → β) (s : Set β) (h : s ⊆ range f) :
    #s ≤ #(f ⁻¹' s) := by
  /-
    α β : Type u
    f : α → β
    s : Set β
    h : HasSubset.Subset s (Set.range f)
    ⊢ LE.le (Cardinal.mk ↑s) (Cardinal.mk ↑(Set.preimage f s))
  -/
  rw [← lift_id #(↑(f ⁻¹' s)), ← lift_id #(↑s)]
  /-
    α β : Type u
    f : α → β
    s : Set β
    h : HasSubset.Subset s (Set.range f)
    ⊢ LE.le (Cardinal.lift.{u, u} (Cardinal.mk ↑s)) (Cardinal.lift.{u, u} (Cardina …
  -/
  exact mk_preimage_of_subset_range_lift f s h
  /-
    🎉 no goals
  -/


theorem mk_subset_ge_of_subset_image_lift {α : Type u} {β : Type v} (f : α → β) {s : Set α}
    {t : Set β} (h : t ⊆ f '' s) : lift.{u} #t ≤ lift.{v} #({ x ∈ s | f x ∈ t } : Set α) := by
  /-
    α : Type u
    β : Type v
    f : α → β
    s : Set α
    t : Set β
    h : HasSubset.Subset t (Set.image f s)
    ⊢ LE.le (Cardinal.lift.{u, v} (Cardinal.mk ↑t)) (Cardinal.lift.{v, u} (Cardina …
  -/
  rw [image_eq_range] at h
  /-
    α : Type u
    β : Type v
    f : α → β
    s : Set α
    t : Set β
    h : HasSubset.Subset t (Set.range fun x => f ↑x)
    ⊢ LE.le (Cardinal.lift.{u, v} (Cardinal.mk ↑t)) (Cardinal.lift.{v, u} (Cardina …
  -/
  convert mk_preimage_of_subset_range_lift _ _ h using 1
  /-
    case h.e'_4
    α : Type u
    β : Type v
    f : α → β
    s : Set α
    t : Set β
    h : HasSubset.Subset t (Set.range fun x => f ↑x)
    ⊢ Eq (Cardinal.lift.{v, u} (Cardinal.mk ↑(setOf fun x => And (Membership.mem s …
  -/
  rw [mk_sep]
  /-
    case h.e'_4
    α : Type u
    β : Type v
    f : α → β
    s : Set α
    t : Set β
    h : HasSubset.Subset t (Set.range fun x => f ↑x)
    ⊢ Eq (Cardinal.lift.{v, u} (Cardinal.mk ↑(setOf fun x => Membership.mem t (f ↑ …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem mk_subset_ge_of_subset_image (f : α → β) {s : Set α} {t : Set β} (h : t ⊆ f '' s) :
    #t ≤ #({ x ∈ s | f x ∈ t } : Set α) := by
  /-
    α β : Type u
    f : α → β
    s : Set α
    t : Set β
    h : HasSubset.Subset t (Set.image f s)
    ⊢ LE.le (Cardinal.mk ↑t) (Cardinal.mk ↑(setOf fun x => And (Membership.mem s x …
  -/
  rw [image_eq_range] at h
  /-
    α β : Type u
    f : α → β
    s : Set α
    t : Set β
    h : HasSubset.Subset t (Set.range fun x => f ↑x)
    ⊢ LE.le (Cardinal.mk ↑t) (Cardinal.mk ↑(setOf fun x => And (Membership.mem s x …
  -/
  convert mk_preimage_of_subset_range _ _ h using 1
  /-
    case h.e'_4
    α β : Type u
    f : α → β
    s : Set α
    t : Set β
    h : HasSubset.Subset t (Set.range fun x => f ↑x)
    ⊢ Eq (Cardinal.mk ↑(setOf fun x => And (Membership.mem s x) (Membership.mem t  …
  -/
  rw [mk_sep]
  /-
    case h.e'_4
    α β : Type u
    f : α → β
    s : Set α
    t : Set β
    h : HasSubset.Subset t (Set.range fun x => f ↑x)
    ⊢ Eq (Cardinal.mk ↑(setOf fun x => Membership.mem t (f ↑x))) (Cardinal.mk ↑(Se …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem le_mk_iff_exists_subset {c : Cardinal} {α : Type u} {s : Set α} :
    c ≤ #s ↔ ∃ p : Set α, p ⊆ s ∧ #p = c := by
  /-
    c : Cardinal.{u}
    α : Type u
    s : Set α
    ⊢ Iff (LE.le c (Cardinal.mk ↑s)) (Exists fun p => And (HasSubset.Subset p s) ( …
  -/
  rw [le_mk_iff_exists_set, ← Subtype.exists_set_subtype]
  /-
    c : Cardinal.{u}
    α : Type u
    s : Set α
    ⊢ Iff (Exists fun p => Eq (Cardinal.mk ↑p) c) (Exists fun s_1 => Eq (Cardinal. …
  -/
  apply exists_congr; intro t; rw [mk_image_eq]; apply Subtype.val_injective
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem two_le_iff : (2 : Cardinal) ≤ #α ↔ ∃ x y : α, x ≠ y := by
  /-
    α : Type u
    ⊢ Iff (LE.le 2 (Cardinal.mk α)) (Exists fun x => Exists fun y => Ne x y)
  -/
  rw [← Nat.cast_two, nat_succ, succ_le_iff, Nat.cast_one, one_lt_iff_nontrivial, nontrivial_iff]
  /-
    🎉 no goals
  -/


theorem two_le_iff' (x : α) : (2 : Cardinal) ≤ #α ↔ ∃ y : α, y ≠ x := by
  /-
    α : Type u
    x : α
    ⊢ Iff (LE.le 2 (Cardinal.mk α)) (Exists fun y => Ne y x)
  -/
  rw [two_le_iff, ← nontrivial_iff, nontrivial_iff_exists_ne x]
  /-
    🎉 no goals
  -/


theorem mk_eq_two_iff : #α = 2 ↔ ∃ x y : α, x ≠ y ∧ ({x, y} : Set α) = univ := by
  classical
  simp only [← @Nat.cast_two Cardinal, mk_eq_nat_iff_finset, Finset.card_eq_two]
  constructor
  · rintro ⟨t, ht, x, y, hne, rfl⟩
    exact ⟨x, y, hne, by simpa using ht⟩
  · rintro ⟨x, y, hne, h⟩
    exact ⟨{x, y}, by simpa using h, x, y, hne, rfl⟩


theorem mk_eq_two_iff' (x : α) : #α = 2 ↔ ∃! y, y ≠ x := by
  /-
    α : Type u
    x : α
    ⊢ Iff (Eq (Cardinal.mk α) 2) (ExistsUnique fun y => Ne y x)
  -/
  rw [mk_eq_two_iff]; constructor
    /-
      case mp
      α : Type u
      x : α
      ⊢ (Exists fun x => Exists fun y => And (Ne x y) (Eq (Insert.insert x (Singleto …
    -/
  · rintro ⟨a, b, hne, h⟩
    /-
      case mp.intro.intro.intro
      α : Type u
      x a b : α
      hne : Ne a b
      h : Eq (Insert.insert a (Singleton.singleton b)) Set.univ
      ⊢ ExistsUnique fun y => Ne y x
    -/
    simp only [eq_univ_iff_forall, mem_insert_iff, mem_singleton_iff] at h
    /-
      case mp.intro.intro.intro
      α : Type u
      x a b : α
      hne : Ne a b
      h : ∀ (x : α), Or (Eq x a) (Eq x b)
      ⊢ ExistsUnique fun y => Ne y x
    -/
    rcases h x with (rfl | rfl)
    /-
      case mp.intro.intro.intro.inl
      α : Type u
      x b : α
      hne : Ne x b
      h : ∀ (x_1 : α), Or (Eq x_1 x) (Eq x_1 b)
      ⊢ ExistsUnique fun y => Ne y x
    -/
    exacts [⟨b, hne.symm, fun z => (h z).resolve_left⟩, ⟨a, hne, fun z => (h z).resolve_right⟩]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      x : α
      ⊢ (ExistsUnique fun y => Ne y x) → Exists fun x => Exists fun y => And (Ne x y …
    -/
  · rintro ⟨y, hne, hy⟩
    /-
      case mpr.intro.intro
      α : Type u
      x y : α
      hne : Ne y x
      hy : ∀ (y_1 : α), (fun y => Ne y x) y_1 → Eq y_1 y
      ⊢ Exists fun x => Exists fun y => And (Ne x y) (Eq (Insert.insert x (Singleton …
    -/
    exact ⟨x, y, hne.symm, eq_univ_of_forall fun z => or_iff_not_imp_left.2 (hy z)⟩
    /-
      🎉 no goals
    -/


theorem exists_not_mem_of_length_lt {α : Type*} (l : List α) (h : ↑l.length < #α) :
    ∃ z : α, z ∉ l := by
  classical
  contrapose! h
  calc
    #α = #(Set.univ : Set α) := mk_univ.symm
    _ ≤ #l.toFinset := mk_le_mk_of_subset fun x _ => List.mem_toFinset.mpr (h x)
    _ = l.toFinset.card := Cardinal.mk_coe_finset
    _ ≤ l.length := Nat.cast_le.mpr (List.toFinset_card_le l)


theorem three_le {α : Type*} (h : 3 ≤ #α) (x : α) (y : α) : ∃ z : α, z ≠ x ∧ z ≠ y := by
  /-
    α : Type u_1
    h : LE.le 3 (Cardinal.mk α)
    x y : α
    ⊢ Exists fun z => And (Ne z x) (Ne z y)
  -/
  have : ↑(3 : ℕ) ≤ #α := by simpa using h
  /-
    α : Type u_1
    h : LE.le 3 (Cardinal.mk α)
    x y : α
    this : LE.le (↑3) (Cardinal.mk α)
    ⊢ Exists fun z => And (Ne z x) (Ne z y)
  -/
  have : ↑(2 : ℕ) < #α := by rwa [← succ_le_iff, ← Cardinal.nat_succ]
  /-
    α : Type u_1
    h : LE.le 3 (Cardinal.mk α)
    x y : α
    this✝ : LE.le (↑3) (Cardinal.mk α)
    this : LT.lt (↑2) (Cardinal.mk α)
    ⊢ Exists fun z => And (Ne z x) (Ne z y)
  -/
  have := exists_not_mem_of_length_lt [x, y] this
  /-
    α : Type u_1
    h : LE.le 3 (Cardinal.mk α)
    x y : α
    this✝¹ : LE.le (↑3) (Cardinal.mk α)
    this✝ : LT.lt (↑2) (Cardinal.mk α)
    this : Exists fun z => Not (Membership.mem (List.cons x (List.cons y List.nil) …
    ⊢ Exists fun z => And (Ne z x) (Ne z y)
  -/
  simpa [not_or] using this
  /-
    🎉 no goals
  -/


/-- The function `a ^< b`, defined as the supremum of `a ^ c` for `c < b`. -/
def powerlt (a b : Cardinal.{u}) : Cardinal.{u} :=
  ⨆ c : Iio b, a ^ (c : Cardinal)


@[inherit_doc]
infixl:80 " ^< " => powerlt


theorem le_powerlt {b c : Cardinal.{u}} (a) (h : c < b) : (a^c) ≤ a ^< b := by
  /-
    b c a : Cardinal.{u}
    h : LT.lt c b
    ⊢ LE.le (HPow.hPow a c) (a.powerlt b)
  -/
  refine le_ciSup (f := fun y : Iio b => a ^ (y : Cardinal)) ?_ ⟨c, h⟩
  /-
    b c a : Cardinal.{u}
    h : LT.lt c b
    ⊢ BddAbove (Set.range fun y => HPow.hPow a ↑y)
  -/
  rw [← image_eq_range]
  /-
    b c a : Cardinal.{u}
    h : LT.lt c b
    ⊢ BddAbove (Set.image (HPow.hPow a) (Set.Iio b))
  -/
  exact bddAbove_image.{u, u} _ bddAbove_Iio
  /-
    🎉 no goals
  -/


theorem powerlt_le {a b c : Cardinal.{u}} : a ^< b ≤ c ↔ ∀ x < b, a ^ x ≤ c := by
  /-
    a b c : Cardinal.{u}
    ⊢ Iff (LE.le (a.powerlt b) c) (∀ (x : Cardinal.{u}), LT.lt x b → LE.le (HPow.h …
  -/
  rw [powerlt, ciSup_le_iff']
    /-
      a b c : Cardinal.{u}
      ⊢ Iff (∀ (i : ↑(Set.Iio b)), LE.le (HPow.hPow a ↑i) c) (∀ (x : Cardinal.{u}),  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h
      a b c : Cardinal.{u}
      ⊢ BddAbove (Set.range fun c => HPow.hPow a ↑c)
    -/
  · rw [← image_eq_range]
    /-
      case h
      a b c : Cardinal.{u}
      ⊢ BddAbove (Set.image (HPow.hPow a) (Set.Iio b))
    -/
    exact bddAbove_image.{u, u} _ bddAbove_Iio
    /-
      🎉 no goals
    -/


theorem powerlt_le_powerlt_left {a b c : Cardinal} (h : b ≤ c) : a ^< b ≤ a ^< c :=
  powerlt_le.2 fun _ hx => le_powerlt a <| hx.trans_le h


theorem powerlt_mono_left (a) : Monotone fun c => a ^< c := fun _ _ => powerlt_le_powerlt_left


theorem powerlt_succ {a b : Cardinal} (h : a ≠ 0) : a ^< succ b = a ^ b :=
  (powerlt_le.2 fun _ h' => power_le_power_left h <| le_of_lt_succ h').antisymm <|
    le_powerlt a (lt_succ b)


theorem powerlt_min {a b c : Cardinal} : a ^< min b c = min (a ^< b) (a ^< c) :=
  (powerlt_mono_left a).map_min


theorem powerlt_max {a b c : Cardinal} : a ^< max b c = max (a ^< b) (a ^< c) :=
  (powerlt_mono_left a).map_max


theorem zero_powerlt {a : Cardinal} (h : a ≠ 0) : 0 ^< a = 1 := by
  /-
    a : Cardinal.{u_1}
    h : Ne a 0
    ⊢ Eq (Cardinal.powerlt 0 a) 1
  -/
  apply (powerlt_le.2 fun c _ => zero_power_le _).antisymm
  /-
    a : Cardinal.{u_1}
    h : Ne a 0
    ⊢ LE.le 1 (Cardinal.powerlt 0 a)
  -/
  rw [← power_zero]
  /-
    a : Cardinal.{u_1}
    h : Ne a 0
    ⊢ LE.le (HPow.hPow ?m.261216 0) (Cardinal.powerlt 0 a)
  -/
  exact le_powerlt 0 (pos_iff_ne_zero.2 h)
  /-
    🎉 no goals
  -/


@[simp]
theorem powerlt_zero {a : Cardinal} : a ^< 0 = 0 := by
  /-
    a : Cardinal.{u_1}
    ⊢ Eq (a.powerlt 0) 0
  -/
  convert Cardinal.iSup_of_empty _
  /-
    case convert_3
    a : Cardinal.{u_1}
    ⊢ IsEmpty ↑(Set.Iio 0)
  -/
  exact Subtype.isEmpty_of_false fun x => mem_Iio.not.mpr (Cardinal.zero_le x).not_lt
  /-
    🎉 no goals
  -/


