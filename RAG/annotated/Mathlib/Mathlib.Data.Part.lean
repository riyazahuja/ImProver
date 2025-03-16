/-- `Part α` is the type of "partial values" of type `α`. It
  is similar to `Option α` except the domain condition can be an
  arbitrary proposition, not necessarily decidable. -/
structure Part.{u} (α : Type u) : Type u where
  /-- The domain of a partial value -/
  Dom : Prop
  /-- Extract a value from a partial value given a proof of `Dom` -/
  get : Dom → α


/-- Convert a `Part α` with a decidable domain to an option -/
def toOption (o : Part α) [Decidable o.Dom] : Option α :=
  if h : Dom o then some (o.get h) else none


@[simp] lemma toOption_isSome (o : Part α) [Decidable o.Dom] : o.toOption.isSome ↔ o.Dom := by
  /-
    α : Type u_1
    o : Part α
    inst✝ : Decidable o.Dom
    ⊢ Iff (Eq o.toOption.isSome Bool.true) o.Dom
  -/
                         /-
                           🎉 no goals
                         -/
  by_cases h : o.Dom <;> simp [h, toOption]
                         /-
                           🎉 no goals
                         -/


@[simp] lemma toOption_eq_none (o : Part α) [Decidable o.Dom] : o.toOption = none ↔ ¬o.Dom := by
  /-
    α : Type u_1
    o : Part α
    inst✝ : Decidable o.Dom
    ⊢ Iff (Eq o.toOption Option.none) (Not o.Dom)
  -/
                         /-
                           🎉 no goals
                         -/
  by_cases h : o.Dom <;> simp [h, toOption]
                         /-
                           🎉 no goals
                         -/


@[deprecated (since := "2024-06-20")] alias toOption_isNone := toOption_eq_none


/-- `Part` extensionality -/
theorem ext' : ∀ {o p : Part α}, (o.Dom ↔ p.Dom) → (∀ h₁ h₂, o.get h₁ = p.get h₂) → o = p
  | ⟨od, o⟩, ⟨pd, p⟩, H1, H2 => by
    /-
      α : Type u_1
      od : Prop
      o : od → α
      pd : Prop
      p : pd → α
      H1 : Iff { Dom := od, get := o }.Dom { Dom := pd, get := p }.Dom
      H2 : ∀ (h₁ : { Dom := od, get := o }.Dom) (h₂ : { Dom := pd, get := p }.Dom),  …
      ⊢ Eq { Dom := od, get := o } { Dom := pd, get := p }
    -/
    have t : od = pd := propext H1
    /-
      α : Type u_1
      od : Prop
      o : od → α
      pd : Prop
      p : pd → α
      H1 : Iff { Dom := od, get := o }.Dom { Dom := pd, get := p }.Dom
      H2 : ∀ (h₁ : { Dom := od, get := o }.Dom) (h₂ : { Dom := pd, get := p }.Dom),  …
      t : Eq od pd
      ⊢ Eq { Dom := od, get := o } { Dom := pd, get := p }
    -/
    cases t; rw [show o = p from funext fun p => H2 p p]
             /-
               🎉 no goals
             -/


/-- `Part` eta expansion -/
@[simp]
theorem eta : ∀ o : Part α, (⟨o.Dom, fun h => o.get h⟩ : Part α) = o
  | ⟨_, _⟩ => rfl


/-- `a ∈ o` means that `o` is defined and equal to `a` -/
protected def Mem (o : Part α) (a : α) : Prop :=
  ∃ h, o.get h = a


instance : Membership α (Part α) :=
  ⟨Part.Mem⟩


theorem mem_eq (a : α) (o : Part α) : (a ∈ o) = ∃ h, o.get h = a :=
  rfl


theorem dom_iff_mem : ∀ {o : Part α}, o.Dom ↔ ∃ y, y ∈ o
  | ⟨_, f⟩ => ⟨fun h => ⟨f h, h, rfl⟩, fun ⟨_, h, rfl⟩ => h⟩


theorem get_mem {o : Part α} (h) : get o h ∈ o :=
  ⟨_, rfl⟩


@[simp]
theorem mem_mk_iff {p : Prop} {o : p → α} {a : α} : a ∈ Part.mk p o ↔ ∃ h, o h = a :=
  Iff.rfl


/-- `Part` extensionality -/
@[ext]
theorem ext {o p : Part α} (H : ∀ a, a ∈ o ↔ a ∈ p) : o = p :=
  (ext' ⟨fun h => ((H _).1 ⟨h, rfl⟩).fst, fun h => ((H _).2 ⟨h, rfl⟩).fst⟩) fun _ _ =>
    ((H _).2 ⟨_, rfl⟩).snd


/-- The `none` value in `Part` has a `False` domain and an empty function. -/
def none : Part α :=
  ⟨False, False.rec⟩


instance : Inhabited (Part α) :=
  ⟨none⟩


@[simp]
theorem not_mem_none (a : α) : a ∉ @none α := fun h => h.fst


/-- The `some a` value in `Part` has a `True` domain and the
  function returns `a`. -/
def some (a : α) : Part α :=
  ⟨True, fun _ => a⟩


@[simp]
theorem some_dom (a : α) : (some a).Dom :=
  trivial


theorem mem_unique : ∀ {a b : α} {o : Part α}, a ∈ o → b ∈ o → a = b
  | _, _, ⟨_, _⟩, ⟨_, rfl⟩, ⟨_, rfl⟩ => rfl


theorem Mem.left_unique : Relator.LeftUnique ((· ∈ ·) : α → Part α → Prop) := fun _ _ _ =>
  mem_unique


theorem get_eq_of_mem {o : Part α} {a} (h : a ∈ o) (h') : get o h' = a :=
  mem_unique ⟨_, rfl⟩ h


protected theorem subsingleton (o : Part α) : Set.Subsingleton { a | a ∈ o } := fun _ ha _ hb =>
  mem_unique ha hb


@[simp]
theorem get_some {a : α} (ha : (some a).Dom) : get (some a) ha = a :=
  rfl


theorem mem_some (a : α) : a ∈ some a :=
  ⟨trivial, rfl⟩


@[simp]
theorem mem_some_iff {a b} : b ∈ (some a : Part α) ↔ b = a :=
  ⟨fun ⟨_, e⟩ => e.symm, fun e => ⟨trivial, e.symm⟩⟩


theorem eq_some_iff {a : α} {o : Part α} : o = some a ↔ a ∈ o :=
  ⟨fun e => e.symm ▸ mem_some _, fun ⟨h, e⟩ => e ▸ ext' (iff_true_intro h) fun _ _ => rfl⟩


theorem eq_none_iff {o : Part α} : o = none ↔ ∀ a, a ∉ o :=
                                                    /-
                                                      α : Type u_1
                                                      o : Part α
                                                      h : ∀ (a : α), Not (Membership.mem o a)
                                                      ⊢ ∀ (a : α), Iff (Membership.mem o a) (Membership.mem Part.none a)
                                                    -/
  ⟨fun e => e.symm ▸ not_mem_none, fun h => ext (by simpa)⟩
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem eq_none_iff' {o : Part α} : o = none ↔ ¬o.Dom :=
  ⟨fun e => e.symm ▸ id, fun h => eq_none_iff.2 fun _ h' => h h'.fst⟩


@[simp]
theorem not_none_dom : ¬(none : Part α).Dom :=
  id


@[simp]
theorem some_ne_none (x : α) : some x ≠ none := by
  /-
    α : Type u_1
    x : α
    ⊢ Ne (Part.some x) Part.none
  -/
  intro h
  /-
    α : Type u_1
    x : α
    h : Eq (Part.some x) Part.none
    ⊢ False
  -/
  exact true_ne_false (congr_arg Dom h)
  /-
    🎉 no goals
  -/


@[simp]
theorem none_ne_some (x : α) : none ≠ some x :=
  (some_ne_none x).symm


theorem ne_none_iff {o : Part α} : o ≠ none ↔ ∃ x, o = some x := by
  /-
    α : Type u_1
    o : Part α
    ⊢ Iff (Ne o Part.none) (Exists fun x => Eq o (Part.some x))
  -/
  constructor
    /-
      case mp
      α : Type u_1
      o : Part α
      ⊢ Ne o Part.none → Exists fun x => Eq o (Part.some x)
    -/
  · rw [Ne, eq_none_iff', not_not]
    /-
      case mp
      α : Type u_1
      o : Part α
      ⊢ o.Dom → Exists fun x => Eq o (Part.some x)
    -/
    exact fun h => ⟨o.get h, eq_some_iff.2 (get_mem h)⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      o : Part α
      ⊢ (Exists fun x => Eq o (Part.some x)) → Ne o Part.none
    -/
  · rintro ⟨x, rfl⟩
    /-
      case mpr.intro
      α : Type u_1
      x : α
      ⊢ Ne (Part.some x) Part.none
    -/
    apply some_ne_none
    /-
      🎉 no goals
    -/


theorem eq_none_or_eq_some (o : Part α) : o = none ∨ ∃ x, o = some x :=
  or_iff_not_imp_left.2 ne_none_iff.1


theorem some_injective : Injective (@Part.some α) := fun _ _ h =>
  congr_fun (eq_of_heq (Part.mk.inj h).2) trivial


@[simp]
theorem some_inj {a b : α} : Part.some a = some b ↔ a = b :=
  some_injective.eq_iff


@[simp]
theorem some_get {a : Part α} (ha : a.Dom) : Part.some (Part.get a ha) = a :=
  Eq.symm (eq_some_iff.2 ⟨ha, rfl⟩)


theorem get_eq_iff_eq_some {a : Part α} {ha : a.Dom} {b : α} : a.get ha = b ↔ a = some b :=
               /-
                 α : Type u_1
                 a : Part α
                 ha : a.Dom
                 b : α
                 h : Eq (a.get ha) b
                 ⊢ Eq a (Part.some b)
               -/
               /-
                 🎉 no goals
               -/
  ⟨fun h => by simp [h.symm], fun h => by simp [h]⟩
                                          /-
                                            🎉 no goals
                                          -/


theorem get_eq_get_of_eq (a : Part α) (ha : a.Dom) {b : Part α} (h : a = b) :
    a.get ha = b.get (h ▸ ha) := by
  /-
    α : Type u_1
    a : Part α
    ha : a.Dom
    b : Part α
    h : Eq a b
    ⊢ Eq (a.get ha) (b.get ⋯)
  -/
  congr
  /-
    🎉 no goals
  -/


theorem get_eq_iff_mem {o : Part α} {a : α} (h : o.Dom) : o.get h = a ↔ a ∈ o :=
  ⟨fun H => ⟨h, H⟩, fun ⟨_, H⟩ => H⟩


theorem eq_get_iff_mem {o : Part α} {a : α} (h : o.Dom) : a = o.get h ↔ a ∈ o :=
  eq_comm.trans (get_eq_iff_mem h)


@[simp]
theorem none_toOption [Decidable (@none α).Dom] : (none : Part α).toOption = Option.none :=
  dif_neg id


@[simp]
theorem some_toOption (a : α) [Decidable (some a).Dom] : (some a).toOption = Option.some a :=
  dif_pos trivial


instance noneDecidable : Decidable (@none α).Dom :=
  instDecidableFalse


instance someDecidable (a : α) : Decidable (some a).Dom :=
  instDecidableTrue


/-- Retrieves the value of `a : Part α` if it exists, and return the provided default value
otherwise. -/
def getOrElse (a : Part α) [Decidable a.Dom] (d : α) :=
  if ha : a.Dom then a.get ha else d


theorem getOrElse_of_dom (a : Part α) (h : a.Dom) [Decidable a.Dom] (d : α) :
    getOrElse a d = a.get h :=
  dif_pos h


theorem getOrElse_of_not_dom (a : Part α) (h : ¬a.Dom) [Decidable a.Dom] (d : α) :
    getOrElse a d = d :=
  dif_neg h


@[simp]
theorem getOrElse_none (d : α) [Decidable (none : Part α).Dom] : getOrElse none d = d :=
  none.getOrElse_of_not_dom not_none_dom d


@[simp]
theorem getOrElse_some (a : α) (d : α) [Decidable (some a).Dom] : getOrElse (some a) d = a :=
  (some a).getOrElse_of_dom (some_dom a) d

-- Porting note: removed `simp`

theorem mem_toOption {o : Part α} [Decidable o.Dom] {a : α} : a ∈ toOption o ↔ a ∈ o := by
  /-
    α : Type u_1
    o : Part α
    inst✝ : Decidable o.Dom
    a : α
    ⊢ Iff (Membership.mem o.toOption a) (Membership.mem o a)
  -/
  unfold toOption
  /-
    α : Type u_1
    o : Part α
    inst✝ : Decidable o.Dom
    a : α
    ⊢ Iff (Membership.mem (dite o.Dom (fun h => Option.some (o.get h)) fun h => Op …
  -/
  by_cases h : o.Dom <;> simp [h]
    /-
      case pos
      α : Type u_1
      o : Part α
      inst✝ : Decidable o.Dom
      a : α
      h : o.Dom
      ⊢ Iff (Eq (o.get ⋯) a) (Membership.mem o a)
    -/
  · exact ⟨fun h => ⟨_, h⟩, fun ⟨_, h⟩ => h⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      o : Part α
      inst✝ : Decidable o.Dom
      a : α
      h : Not o.Dom
      ⊢ Not (Membership.mem o a)
    -/
  · exact mt Exists.fst h
    /-
      🎉 no goals
    -/


@[simp]
theorem toOption_eq_some_iff {o : Part α} [Decidable o.Dom] {a : α} :
    toOption o = Option.some a ↔ a ∈ o := by
  /-
    α : Type u_1
    o : Part α
    inst✝ : Decidable o.Dom
    a : α
    ⊢ Iff (Eq o.toOption (Option.some a)) (Membership.mem o a)
  -/
  rw [← Option.mem_def, mem_toOption]
  /-
    🎉 no goals
  -/


protected theorem Dom.toOption {o : Part α} [Decidable o.Dom] (h : o.Dom) : o.toOption = o.get h :=
  dif_pos h


theorem toOption_eq_none_iff {a : Part α} [Decidable a.Dom] : a.toOption = Option.none ↔ ¬a.Dom :=
  Ne.dite_eq_right_iff fun _ => Option.some_ne_none _

/- Porting TODO: Removed `simp`. Maybe add `@[simp]` later if `@[simp]` is taken off definition of
`Option.elim` -/

theorem elim_toOption {α β : Type*} (a : Part α) [Decidable a.Dom] (b : β) (f : α → β) :
    a.toOption.elim b f = if h : a.Dom then f (a.get h) else b := by
  /-
    α : Type u_4
    β : Type u_5
    a : Part α
    inst✝ : Decidable a.Dom
    b : β
    f : α → β
    ⊢ Eq (a.toOption.elim b f) (dite a.Dom (fun h => f (a.get h)) fun h => b)
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_4
      β : Type u_5
      a : Part α
      inst✝ : Decidable a.Dom
      b : β
      f : α → β
      h : a.Dom
      ⊢ Eq (a.toOption.elim b f) (f (a.get h))
    -/
  · rw [h.toOption]
    /-
      case pos
      α : Type u_4
      β : Type u_5
      a : Part α
      inst✝ : Decidable a.Dom
      b : β
      f : α → β
      h : a.Dom
      ⊢ Eq ((Option.some (a.get h)).elim b f) (f (a.get h))
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_4
      β : Type u_5
      a : Part α
      inst✝ : Decidable a.Dom
      b : β
      f : α → β
      h : Not a.Dom
      ⊢ Eq (a.toOption.elim b f) b
    -/
  · rw [Part.toOption_eq_none_iff.2 h]
    /-
      case neg
      α : Type u_4
      β : Type u_5
      a : Part α
      inst✝ : Decidable a.Dom
      b : β
      f : α → β
      h : Not a.Dom
      ⊢ Eq (Option.none.elim b f) b
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- Converts an `Option α` into a `Part α`. -/
@[coe]
def ofOption : Option α → Part α
  | Option.none => none
  | Option.some a => some a


@[simp]
theorem mem_ofOption {a : α} : ∀ {o : Option α}, a ∈ ofOption o ↔ a ∈ o
  | Option.none => ⟨fun h => h.fst.elim, fun h => Option.noConfusion h⟩
  | Option.some _ => ⟨fun h => congr_arg Option.some h.snd, fun h => ⟨trivial, Option.some.inj h⟩⟩


@[simp]
theorem ofOption_dom {α} : ∀ o : Option α, (ofOption o).Dom ↔ o.isSome
                      /-
                        α : Type u_4
                        ⊢ Iff (↑Option.none).Dom (Eq Option.none.isSome Bool.true)
                      -/
  | Option.none => by simp [ofOption, none]
                      /-
                        🎉 no goals
                      -/
                        /-
                          α : Type u_4
                          a : α
                          ⊢ Iff (↑(Option.some a)).Dom (Eq (Option.some a).isSome Bool.true)
                        -/
  | Option.some a => by simp [ofOption]
                        /-
                          🎉 no goals
                        -/


theorem ofOption_eq_get {α} (o : Option α) : ofOption o = ⟨_, @Option.get _ o⟩ :=
  Part.ext' (ofOption_dom o) fun h₁ h₂ => by
    /-
      α : Type u_4
      o : Option α
      h₁ : (↑o).Dom
      h₂ : { Dom := Eq o.isSome Bool.true, get := o.get }.Dom
      ⊢ Eq ((↑o).get h₁) ({ Dom := Eq o.isSome Bool.true, get := o.get }.get h₂)
    -/
    cases o
      /-
        case none
        α : Type u_4
        h₁ : (↑Option.none).Dom
        h₂ : { Dom := Eq Option.none.isSome Bool.true, get := Option.none.get }.Dom
        ⊢ Eq ((↑Option.none).get h₁) ({ Dom := Eq Option.none.isSome Bool.true, get := …
      -/
    · simp at h₂
      /-
        🎉 no goals
      -/
      /-
        case some
        α : Type u_4
        val✝ : α
        h₁ : (↑(Option.some val✝)).Dom
        h₂ : { Dom := Eq (Option.some val✝).isSome Bool.true, get := (Option.some val✝ …
        ⊢ Eq ((↑(Option.some val✝)).get h₁) ({ Dom := Eq (Option.some val✝).isSome Boo …
      -/
    · rfl
      /-
        🎉 no goals
      -/


instance : Coe (Option α) (Part α) :=
  ⟨ofOption⟩


theorem mem_coe {a : α} {o : Option α} : a ∈ (o : Part α) ↔ a ∈ o :=
  mem_ofOption


@[simp]
theorem coe_none : (@Option.none α : Part α) = none :=
  rfl


@[simp]
theorem coe_some (a : α) : (Option.some a : Part α) = some a :=
  rfl


@[elab_as_elim]
protected theorem induction_on {P : Part α → Prop} (a : Part α) (hnone : P none)
    (hsome : ∀ a : α, P (some a)) : P a :=
  (Classical.em a.Dom).elim (fun h => Part.some_get h ▸ hsome _) fun h =>
    (eq_none_iff'.2 h).symm ▸ hnone


instance ofOptionDecidable : ∀ o : Option α, Decidable (ofOption o).Dom
  | Option.none => Part.noneDecidable
  | Option.some a => Part.someDecidable a


@[simp]
                                                                     /-
                                                                       α : Type u_1
                                                                       o : Option α
                                                                       ⊢ Eq (↑o).toOption o
                                                                     -/
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
theorem to_ofOption (o : Option α) : toOption (ofOption o) = o := by cases o <;> rfl
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


@[simp]
theorem of_toOption (o : Part α) [Decidable o.Dom] : ofOption (toOption o) = o :=
  ext fun _ => mem_ofOption.trans mem_toOption


/-- `Part α` is (classically) equivalent to `Option α`. -/
noncomputable def equivOption : Part α ≃ Option α :=
  haveI := Classical.dec
  ⟨fun o => toOption o, ofOption, fun o => of_toOption o, fun o =>
                 /-
                   α : Type u_1
                   β : Type u_2
                   γ : Type u_3
                   this : (p : Prop) → Decidable p
                   o : Option α
                   ⊢ Eq ((fun o => o.toOption) ↑o) (↑o).toOption
                 -/
    Eq.trans (by dsimp; congr) (to_ofOption o)⟩
                        /-
                          🎉 no goals
                        -/


/-- We give `Part α` the order where everything is greater than `none`. -/
instance : PartialOrder (Part
        α) where
  le x y := ∀ i, i ∈ x → i ∈ y
  le_refl _ _ := id
  le_trans _ _ _ f g _ := g _ ∘ f _
  le_antisymm _ _ f g := Part.ext fun _ => ⟨f _, g _⟩


instance : OrderBot (Part α) where
  bot := none
               /-
                 α : Type u_1
                 β : Type u_2
                 γ : Type u_3
                 ⊢ ∀ (a : Part α), LE.le Bot.bot a
               -/
  bot_le := by rintro x _ ⟨⟨_⟩, _⟩
               /-
                 🎉 no goals
               -/


theorem le_total_of_le_of_le {x y : Part α} (z : Part α) (hx : x ≤ z) (hy : y ≤ z) :
    x ≤ y ∨ y ≤ x := by
  /-
    α : Type u_1
    x y z : Part α
    hx : LE.le x z
    hy : LE.le y z
    ⊢ Or (LE.le x y) (LE.le y x)
  -/
  rcases Part.eq_none_or_eq_some x with (h | ⟨b, h₀⟩)
    /-
      case inl
      α : Type u_1
      x y z : Part α
      hx : LE.le x z
      hy : LE.le y z
      h : Eq x Part.none
      ⊢ Or (LE.le x y) (LE.le y x)
    -/
  · rw [h]
    /-
      case inl
      α : Type u_1
      x y z : Part α
      hx : LE.le x z
      hy : LE.le y z
      h : Eq x Part.none
      ⊢ Or (LE.le Part.none y) (LE.le y Part.none)
    -/
    left
    /-
      case inl.h
      α : Type u_1
      x y z : Part α
      hx : LE.le x z
      hy : LE.le y z
      h : Eq x Part.none
      ⊢ LE.le Part.none y
    -/
    apply OrderBot.bot_le _
    /-
      🎉 no goals
    -/
  /-
    case inr.intro
    α : Type u_1
    x y z : Part α
    hx : LE.le x z
    hy : LE.le y z
    b : α
    h₀ : Eq x (Part.some b)
    ⊢ Or (LE.le x y) (LE.le y x)
  -/
  right; intro b' h₁
  /-
    case inr.intro.h
    α : Type u_1
    x y z : Part α
    hx : LE.le x z
    hy : LE.le y z
    b : α
    h₀ : Eq x (Part.some b)
    b' : α
    h₁ : Membership.mem y b'
    ⊢ Membership.mem x b'
  -/
  rw [Part.eq_some_iff] at h₀
  /-
    case inr.intro.h
    α : Type u_1
    x y z : Part α
    hx : LE.le x z
    hy : LE.le y z
    b : α
    h₀ : Membership.mem x b
    b' : α
    h₁ : Membership.mem y b'
    ⊢ Membership.mem x b'
  -/
  have hx := hx _ h₀; have hy := hy _ h₁
  /-
    case inr.intro.h
    α : Type u_1
    x y z : Part α
    hx✝ : LE.le x z
    hy✝ : LE.le y z
    b : α
    h₀ : Membership.mem x b
    b' : α
    h₁ : Membership.mem y b'
    hx : Membership.mem z b
    hy : Membership.mem z b'
    ⊢ Membership.mem x b'
  -/
  have hx := Part.mem_unique hx hy; subst hx
  /-
    case inr.intro.h
    α : Type u_1
    x y z : Part α
    hx✝ : LE.le x z
    hy✝ : LE.le y z
    b : α
    h₀ : Membership.mem x b
    hx : Membership.mem z b
    h₁ : Membership.mem y b
    hy : Membership.mem z b
    ⊢ Membership.mem x b
  -/
  exact h₀
  /-
    🎉 no goals
  -/


/-- `assert p f` is a bind-like operation which appends an additional condition
  `p` to the domain and uses `f` to produce the value. -/
def assert (p : Prop) (f : p → Part α) : Part α :=
  ⟨∃ h : p, (f h).Dom, fun ha => (f ha.fst).get ha.snd⟩


/-- The bind operation has value `g (f.get)`, and is defined when all the
  parts are defined. -/
protected def bind (f : Part α) (g : α → Part β) : Part β :=
  assert (Dom f) fun b => g (f.get b)


/-- The map operation for `Part` just maps the value and maintains the same domain. -/
@[simps]
def map (f : α → β) (o : Part α) : Part β :=
  ⟨o.Dom, f ∘ o.get⟩


theorem mem_map (f : α → β) {o : Part α} : ∀ {a}, a ∈ o → f a ∈ map f o
  | _, ⟨_, rfl⟩ => ⟨_, rfl⟩


@[simp]
theorem mem_map_iff (f : α → β) {o : Part α} {b} : b ∈ map f o ↔ ∃ a ∈ o, f a = b :=
  ⟨fun hb => match b, hb with
    | _, ⟨_, rfl⟩ => ⟨_, ⟨_, rfl⟩, rfl⟩,
    fun ⟨_, h₁, h₂⟩ => h₂ ▸ mem_map f h₁⟩


@[simp]
theorem map_none (f : α → β) : map f none = none :=
                            /-
                              α : Type u_1
                              β : Type u_2
                              f : α → β
                              a : β
                              ⊢ Not (Membership.mem (Part.map f Part.none) a)
                            -/
  eq_none_iff.2 fun a => by simp
                            /-
                              🎉 no goals
                            -/


@[simp]
theorem map_some (f : α → β) (a : α) : map f (some a) = some (f a) :=
  eq_some_iff.2 <| mem_map f <| mem_some _


theorem mem_assert {p : Prop} {f : p → Part α} : ∀ {a} (h : p), a ∈ f h → a ∈ assert p f
  | _, x, ⟨h, rfl⟩ => ⟨⟨x, h⟩, rfl⟩


@[simp]
theorem mem_assert_iff {p : Prop} {f : p → Part α} {a} : a ∈ assert p f ↔ ∃ h : p, a ∈ f h :=
  ⟨fun ha => match a, ha with
    | _, ⟨_, rfl⟩ => ⟨_, ⟨_, rfl⟩⟩,
    fun ⟨_, h⟩ => mem_assert _ h⟩


theorem assert_pos {p : Prop} {f : p → Part α} (h : p) : assert p f = f h := by
  /-
    α : Type u_1
    p : Prop
    f : p → Part α
    h : p
    ⊢ Eq (Part.assert p f) (f h)
  -/
  dsimp [assert]
  /-
    α : Type u_1
    p : Prop
    f : p → Part α
    h : p
    ⊢ Eq { Dom := Exists fun h => (f h).Dom, get := fun ha => (f ⋯).get ⋯ } (f h)
  -/
  cases h' : f h
  /-
    case mk
    α : Type u_1
    p : Prop
    f : p → Part α
    h : p
    Dom✝ : Prop
    get✝ : Dom✝ → α
    h' : Eq (f h) { Dom := Dom✝, get := get✝ }
    ⊢ Eq { Dom := Exists fun h => (f h).Dom, get := fun ha => (f ⋯).get ⋯ } { Dom  …
  -/
  simp only [h', mk.injEq, h, exists_prop_of_true, true_and]
  /-
    case mk
    α : Type u_1
    p : Prop
    f : p → Part α
    h : p
    Dom✝ : Prop
    get✝ : Dom✝ → α
    h' : Eq (f h) { Dom := Dom✝, get := get✝ }
    ⊢ HEq (fun ha => get✝ ⋯) get✝
  -/
  apply Function.hfunext
    /-
      case mk.hα
      α : Type u_1
      p : Prop
      f : p → Part α
      h : p
      Dom✝ : Prop
      get✝ : Dom✝ → α
      h' : Eq (f h) { Dom := Dom✝, get := get✝ }
      ⊢ Eq (Exists fun h => (f h).Dom) Dom✝
    -/
  · simp only [h, h', exists_prop_of_true]
    /-
      🎉 no goals
    -/
    /-
      case mk.h
      α : Type u_1
      p : Prop
      f : p → Part α
      h : p
      Dom✝ : Prop
      get✝ : Dom✝ → α
      h' : Eq (f h) { Dom := Dom✝, get := get✝ }
      ⊢ ∀ (a : Exists fun h => (f h).Dom) (a' : Dom✝), HEq a a' → HEq (get✝ ⋯) (get✝ …
    -/
  · aesop
    /-
      🎉 no goals
    -/


theorem assert_neg {p : Prop} {f : p → Part α} (h : ¬p) : assert p f = none := by
  /-
    α : Type u_1
    p : Prop
    f : p → Part α
    h : Not p
    ⊢ Eq (Part.assert p f) Part.none
  -/
  dsimp [assert, none]; congr
    /-
      case h.e_2
      α : Type u_1
      p : Prop
      f : p → Part α
      h : Not p
      ⊢ Eq (Exists fun h => (f h).Dom) False
    -/
  · simp only [h, not_false_iff, exists_prop_of_false]
    /-
      🎉 no goals
    -/
    /-
      case h.e_3
      α : Type u_1
      p : Prop
      f : p → Part α
      h : Not p
      ⊢ HEq (fun ha => (f ⋯).get ⋯) fun t => False.rec (fun x => α) t
    -/
  · apply Function.hfunext
      /-
        case h.e_3.hα
        α : Type u_1
        p : Prop
        f : p → Part α
        h : Not p
        ⊢ Eq (Exists fun h => (f h).Dom) False
      -/
    · simp only [h, not_false_iff, exists_prop_of_false]
      /-
        🎉 no goals
      -/
    /-
      case h.e_3.h
      α : Type u_1
      p : Prop
      f : p → Part α
      h : Not p
      ⊢ ∀ (a : Exists fun h => (f h).Dom) (a' : False), HEq a a' → HEq ((f ⋯).get ⋯) …
    -/
    simp at *
    /-
      🎉 no goals
    -/


theorem mem_bind {f : Part α} {g : α → Part β} : ∀ {a b}, a ∈ f → b ∈ g a → b ∈ f.bind g
  | _, _, ⟨h, rfl⟩, ⟨h₂, rfl⟩ => ⟨⟨h, h₂⟩, rfl⟩


@[simp]
theorem mem_bind_iff {f : Part α} {g : α → Part β} {b} : b ∈ f.bind g ↔ ∃ a ∈ f, b ∈ g a :=
  ⟨fun hb => match b, hb with
    | _, ⟨⟨_, _⟩, rfl⟩ => ⟨_, ⟨_, rfl⟩, ⟨_, rfl⟩⟩,
    fun ⟨_, h₁, h₂⟩ => mem_bind h₁ h₂⟩


protected theorem Dom.bind {o : Part α} (h : o.Dom) (f : α → Part β) : o.bind f = f (o.get h) := by
  /-
    α : Type u_1
    β : Type u_2
    o : Part α
    h : o.Dom
    f : α → Part β
    ⊢ Eq (o.bind f) (f (o.get h))
  -/
  ext b
  /-
    case H
    α : Type u_1
    β : Type u_2
    o : Part α
    h : o.Dom
    f : α → Part β
    b : β
    ⊢ Iff (Membership.mem (o.bind f) b) (Membership.mem (f (o.get h)) b)
  -/
  simp only [Part.mem_bind_iff, exists_prop]
  /-
    case H
    α : Type u_1
    β : Type u_2
    o : Part α
    h : o.Dom
    f : α → Part β
    b : β
    ⊢ Iff (Exists fun a => And (Membership.mem o a) (Membership.mem (f a) b)) (Mem …
  -/
  refine ⟨?_, fun hb => ⟨o.get h, Part.get_mem _, hb⟩⟩
  /-
    case H
    α : Type u_1
    β : Type u_2
    o : Part α
    h : o.Dom
    f : α → Part β
    b : β
    ⊢ (Exists fun a => And (Membership.mem o a) (Membership.mem (f a) b)) → Member …
  -/
  rintro ⟨a, ha, hb⟩
  /-
    case H.intro.intro
    α : Type u_1
    β : Type u_2
    o : Part α
    h : o.Dom
    f : α → Part β
    b : β
    a : α
    ha : Membership.mem o a
    hb : Membership.mem (f a) b
    ⊢ Membership.mem (f (o.get h)) b
  -/
  rwa [Part.get_eq_of_mem ha]
  /-
    🎉 no goals
  -/


theorem Dom.of_bind {f : α → Part β} {a : Part α} (h : (a.bind f).Dom) : a.Dom :=
  h.1


@[simp]
theorem bind_none (f : α → Part β) : none.bind f = none :=
                            /-
                              α : Type u_1
                              β : Type u_2
                              f : α → Part β
                              a : β
                              ⊢ Not (Membership.mem (Part.none.bind f) a)
                            -/
  eq_none_iff.2 fun a => by simp
                            /-
                              🎉 no goals
                            -/


@[simp]
theorem bind_some (a : α) (f : α → Part β) : (some a).bind f = f a :=
            /-
              α : Type u_1
              β : Type u_2
              a : α
              f : α → Part β
              ⊢ ∀ (a_1 : β), Iff (Membership.mem ((Part.some a).bind f) a_1) (Membership.mem …
            -/
  ext <| by simp
            /-
              🎉 no goals
            -/


theorem bind_of_mem {o : Part α} {a : α} (h : a ∈ o) (f : α → Part β) : o.bind f = f a := by
  /-
    α : Type u_1
    β : Type u_2
    o : Part α
    a : α
    h : Membership.mem o a
    f : α → Part β
    ⊢ Eq (o.bind f) (f a)
  -/
  rw [eq_some_iff.2 h, bind_some]
  /-
    🎉 no goals
  -/


theorem bind_some_eq_map (f : α → β) (x : Part α) : x.bind (some ∘ f) = map f x :=
            /-
              α : Type u_1
              β : Type u_2
              f : α → β
              x : Part α
              ⊢ ∀ (a : β), Iff (Membership.mem (x.bind (Function.comp Part.some f)) a) (Memb …
            -/
  ext <| by simp [eq_comm]
            /-
              🎉 no goals
            -/


theorem bind_toOption (f : α → Part β) (o : Part α) [Decidable o.Dom] [∀ a, Decidable (f a).Dom]
    [Decidable (o.bind f).Dom] :
    (o.bind f).toOption = o.toOption.elim Option.none fun a => (f a).toOption := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → Part β
    o : Part α
    inst✝² : Decidable o.Dom
    inst✝¹ : (a : α) → Decidable (f a).Dom
    inst✝ : Decidable (o.bind f).Dom
    ⊢ Eq (o.bind f).toOption (o.toOption.elim Option.none fun a => (f a).toOption)
  -/
  by_cases h : o.Dom
    /-
      case pos
      α : Type u_1
      β : Type u_2
      f : α → Part β
      o : Part α
      inst✝² : Decidable o.Dom
      inst✝¹ : (a : α) → Decidable (f a).Dom
      inst✝ : Decidable (o.bind f).Dom
      h : o.Dom
      ⊢ Eq (o.bind f).toOption (o.toOption.elim Option.none fun a => (f a).toOption)
    -/
  · simp_rw [h.toOption, h.bind]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      f : α → Part β
      o : Part α
      inst✝² : Decidable o.Dom
      inst✝¹ : (a : α) → Decidable (f a).Dom
      inst✝ : Decidable (o.bind f).Dom
      h : o.Dom
      ⊢ Eq (f (o.get h)).toOption ((Option.some (o.get h)).elim Option.none fun a => …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      f : α → Part β
      o : Part α
      inst✝² : Decidable o.Dom
      inst✝¹ : (a : α) → Decidable (f a).Dom
      inst✝ : Decidable (o.bind f).Dom
      h : Not o.Dom
      ⊢ Eq (o.bind f).toOption (o.toOption.elim Option.none fun a => (f a).toOption)
    -/
  · rw [Part.toOption_eq_none_iff.2 h]
    /-
      case neg
      α : Type u_1
      β : Type u_2
      f : α → Part β
      o : Part α
      inst✝² : Decidable o.Dom
      inst✝¹ : (a : α) → Decidable (f a).Dom
      inst✝ : Decidable (o.bind f).Dom
      h : Not o.Dom
      ⊢ Eq (o.bind f).toOption (Option.none.elim Option.none fun a => (f a).toOption)
    -/
    exact Part.toOption_eq_none_iff.2 fun ho => h ho.of_bind
    /-
      🎉 no goals
    -/


theorem bind_assoc {γ} (f : Part α) (g : α → Part β) (k : β → Part γ) :
    (f.bind g).bind k = f.bind fun x => (g x).bind k :=
  ext fun a => by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_4
      f : Part α
      g : α → Part β
      k : β → Part γ
      a : γ
      ⊢ Iff (Membership.mem ((f.bind g).bind k) a) (Membership.mem (f.bind fun x =>  …
    -/
    simp only [mem_bind_iff]
    exact ⟨fun ⟨_, ⟨_, h₁, h₂⟩, h₃⟩ => ⟨_, h₁, _, h₂, h₃⟩,
           fun ⟨_, h₁, _, h₂, h₃⟩ => ⟨_, ⟨_, h₁, h₂⟩, h₃⟩⟩


@[simp]
theorem bind_map {γ} (f : α → β) (x) (g : β → Part γ) :
                                                     /-
                                                       α : Type u_1
                                                       β : Type u_2
                                                       γ : Type u_4
                                                       f : α → β
                                                       x : Part α
                                                       g : β → Part γ
                                                       ⊢ Eq ((Part.map f x).bind g) (x.bind fun y => g (f y))
                                                     -/
    (map f x).bind g = x.bind fun y => g (f y) := by rw [← bind_some_eq_map, bind_assoc]; simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


@[simp]
theorem map_bind {γ} (f : α → Part β) (x : Part α) (g : β → γ) :
    map g (x.bind f) = x.bind fun y => map g (f y) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_4
    f : α → Part β
    x : Part α
    g : β → γ
    ⊢ Eq (Part.map g (x.bind f)) (x.bind fun y => Part.map g (f y))
  -/
  rw [← bind_some_eq_map, bind_assoc]; simp [bind_some_eq_map]
                                       /-
                                         🎉 no goals
                                       -/


theorem map_map (g : β → γ) (f : α → β) (o : Part α) : map g (map f o) = map (g ∘ f) o := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    g : β → γ
    f : α → β
    o : Part α
    ⊢ Eq (Part.map g (Part.map f o)) (Part.map (Function.comp g f) o)
  -/
  erw [← bind_some_eq_map, bind_map, bind_some_eq_map]
  /-
    🎉 no goals
  -/


instance : Monad Part where
  pure := @some
  map := @map
  bind := @Part.bind


instance : LawfulMonad
      Part where
  bind_pure_comp := @bind_some_eq_map
                 /-
                   α : Type u_1
                   β : Type u_2
                   γ : Type u_3
                   α✝ : Type u_4
                   f : Part α✝
                   ⊢ Eq (Functor.map id f) f
                 -/
  id_map f := by cases f; rfl
                          /-
                            🎉 no goals
                          -/
                  /-
                    α : Type u_1
                    β : Type u_2
                    γ : Type u_3
                    ⊢ ∀ {α β : Type u_4}, Eq Functor.mapConst (Function.comp Functor.map (Function …
                  -/
  pure_bind := @bind_some
                  /-
                    🎉 no goals
                  -/
  bind_assoc := @bind_assoc
  map_const := by simp [Functor.mapConst, Functor.map]
  --Porting TODO : In Lean3 these were automatic by a tactic
  seqLeft_eq x y := ext'
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          α✝ β✝ : Type u_4
          x : Part α✝
          y : Part β✝
          ⊢ Iff (SeqLeft.seqLeft x fun x => y).Dom (Seq.seq (Functor.map (Function.const …
        -/
    (by simp [SeqLeft.seqLeft, Part.bind, assert, Seq.seq, const, (· <$> ·), and_comm])
        /-
          🎉 no goals
        -/
    (fun _ _ => rfl)
  seqRight_eq x y := ext'
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          α✝ β✝ : Type u_4
          x : Part α✝
          y : Part β✝
          ⊢ Iff (SeqRight.seqRight x fun x => y).Dom (Seq.seq (Functor.map (Function.con …
        -/
    (by simp [SeqRight.seqRight, Part.bind, assert, Seq.seq, const, (· <$> ·), and_comm])
        /-
          🎉 no goals
        -/
    (fun _ _ => rfl)
  pure_seq x y := ext'
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          α✝ β✝ : Type u_4
          x : α✝ → β✝
          y : Part α✝
          ⊢ Iff (Seq.seq (Pure.pure x) fun x => y).Dom (Functor.map x y).Dom
        -/
    (by simp [Seq.seq, Part.bind, assert, (· <$> ·), pure])
        /-
          🎉 no goals
        -/
    (fun _ _ => rfl)
  bind_map x y := ext'
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          α✝ β✝ : Type u_4
          x : Part (α✝ → β✝)
          y : Part α✝
          ⊢ Iff (Bind.bind x fun x => Functor.map x y).Dom (Seq.seq x fun x => y).Dom
        -/
    (by simp [(· >>= ·), Part.bind, assert, Seq.seq, get, (· <$> ·)] )
        /-
          🎉 no goals
        -/
    (fun _ _ => rfl)


theorem map_id' {f : α → α} (H : ∀ x : α, f x = x) (o) : map f o = o := by
  /-
    α : Type u_1
    f : α → α
    H : ∀ (x : α), Eq (f x) x
    o : Part α
    ⊢ Eq (Part.map f o) o
  -/
  rw [show f = id from funext H]; exact id_map o
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem bind_some_right (x : Part α) : x.bind some = x := by
  /-
    α : Type u_1
    x : Part α
    ⊢ Eq (x.bind Part.some) x
  -/
  erw [bind_some_eq_map]; simp [map_id']
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem pure_eq_some (a : α) : pure a = some a :=
  rfl


@[simp]
theorem ret_eq_some (a : α) : (return a : Part α) = some a :=
  rfl


@[simp]
theorem map_eq_map {α β} (f : α → β) (o : Part α) : f <$> o = map f o :=
  rfl


@[simp]
theorem bind_eq_bind {α β} (f : Part α) (g : α → Part β) : f >>= g = f.bind g :=
  rfl


theorem bind_le {α} (x : Part α) (f : α → Part β) (y : Part β) :
    x >>= f ≤ y ↔ ∀ a, a ∈ x → f a ≤ y := by
  /-
    β α : Type u_2
    x : Part α
    f : α → Part β
    y : Part β
    ⊢ Iff (LE.le (Bind.bind x f) y) (∀ (a : α), Membership.mem x a → LE.le (f a) y)
  -/
  constructor <;> intro h
    /-
      case mp
      β α : Type u_2
      x : Part α
      f : α → Part β
      y : Part β
      h : LE.le (Bind.bind x f) y
      ⊢ ∀ (a : α), Membership.mem x a → LE.le (f a) y
    -/
  · intro a h' b
    /-
      case mp
      β α : Type u_2
      x : Part α
      f : α → Part β
      y : Part β
      h : LE.le (Bind.bind x f) y
      a : α
      h' : Membership.mem x a
      b : β
      ⊢ Membership.mem (f a) b → Membership.mem y b
    -/
    have h := h b
    /-
      case mp
      β α : Type u_2
      x : Part α
      f : α → Part β
      y : Part β
      h✝ : LE.le (Bind.bind x f) y
      a : α
      h' : Membership.mem x a
      b : β
      h : Membership.mem (Bind.bind x f) b → Membership.mem y b
      ⊢ Membership.mem (f a) b → Membership.mem y b
    -/
    simp only [and_imp, exists_prop, bind_eq_bind, mem_bind_iff, exists_imp] at h
    /-
      case mp
      β α : Type u_2
      x : Part α
      f : α → Part β
      y : Part β
      h✝ : LE.le (Bind.bind x f) y
      a : α
      h' : Membership.mem x a
      b : β
      h : ∀ (x_1 : α), Membership.mem x x_1 → Membership.mem (f x_1) b → Membership. …
      ⊢ Membership.mem (f a) b → Membership.mem y b
    -/
    apply h _ h'
    /-
      🎉 no goals
    -/
    /-
      case mpr
      β α : Type u_2
      x : Part α
      f : α → Part β
      y : Part β
      h : ∀ (a : α), Membership.mem x a → LE.le (f a) y
      ⊢ LE.le (Bind.bind x f) y
    -/
  · intro b h'
    /-
      case mpr
      β α : Type u_2
      x : Part α
      f : α → Part β
      y : Part β
      h : ∀ (a : α), Membership.mem x a → LE.le (f a) y
      b : β
      h' : Membership.mem (Bind.bind x f) b
      ⊢ Membership.mem y b
    -/
    simp only [exists_prop, bind_eq_bind, mem_bind_iff] at h'
    /-
      case mpr
      β α : Type u_2
      x : Part α
      f : α → Part β
      y : Part β
      h : ∀ (a : α), Membership.mem x a → LE.le (f a) y
      b : β
      h' : Exists fun a => And (Membership.mem x a) (Membership.mem (f a) b)
      ⊢ Membership.mem y b
    -/
    rcases h' with ⟨a, h₀, h₁⟩
    /-
      case mpr.intro.intro
      β α : Type u_2
      x : Part α
      f : α → Part β
      y : Part β
      h : ∀ (a : α), Membership.mem x a → LE.le (f a) y
      b : β
      a : α
      h₀ : Membership.mem x a
      h₁ : Membership.mem (f a) b
      ⊢ Membership.mem y b
    -/
    apply h _ h₀ _ h₁
    /-
      🎉 no goals
    -/

-- Porting note: No MonadFail in Lean4 yet
-- instance : MonadFail Part :=
--   { Part.monad with fail := fun _ _ => none }


/-- `restrict p o h` replaces the domain of `o` with `p`, and is well defined when
  `p` implies `o` is defined. -/
def restrict (p : Prop) (o : Part α) (H : p → o.Dom) : Part α :=
  ⟨p, fun h => o.get (H h)⟩


@[simp]
theorem mem_restrict (p : Prop) (o : Part α) (h : p → o.Dom) (a : α) :
    a ∈ restrict p o h ↔ p ∧ a ∈ o := by
  /-
    α : Type u_1
    p : Prop
    o : Part α
    h : p → o.Dom
    a : α
    ⊢ Iff (Membership.mem (Part.restrict p o h) a) (And p (Membership.mem o a))
  -/
  dsimp [restrict, mem_eq]; constructor
    /-
      case mp
      α : Type u_1
      p : Prop
      o : Part α
      h : p → o.Dom
      a : α
      ⊢ (Exists fun h_1 => Eq (o.get ⋯) a) → And p (Exists fun h => Eq (o.get h) a)
    -/
  · rintro ⟨h₀, h₁⟩
    /-
      case mp.intro
      α : Type u_1
      p : Prop
      o : Part α
      h : p → o.Dom
      a : α
      h₀ : p
      h₁ : Eq (o.get ⋯) a
      ⊢ And p (Exists fun h => Eq (o.get h) a)
    -/
    exact ⟨h₀, ⟨_, h₁⟩⟩
    /-
      🎉 no goals
    -/
  /-
    case mpr
    α : Type u_1
    p : Prop
    o : Part α
    h : p → o.Dom
    a : α
    ⊢ And p (Exists fun h => Eq (o.get h) a) → Exists fun h_1 => Eq (o.get ⋯) a
  -/
  rintro ⟨h₀, _, h₂⟩; exact ⟨h₀, h₂⟩
                      /-
                        🎉 no goals
                      -/


/-- `unwrap o` gets the value at `o`, ignoring the condition. This function is unsound. -/
unsafe def unwrap (o : Part α) : α :=
  o.get lcProof


theorem assert_defined {p : Prop} {f : p → Part α} : ∀ h : p, (f h).Dom → (assert p f).Dom :=
  Exists.intro


theorem bind_defined {f : Part α} {g : α → Part β} :
    ∀ h : f.Dom, (g (f.get h)).Dom → (f.bind g).Dom :=
  assert_defined


@[simp]
theorem bind_dom {f : Part α} {g : α → Part β} : (f.bind g).Dom ↔ ∃ h : f.Dom, (g (f.get h)).Dom :=
  Iff.rfl


@[to_additive]
instance [One α] : One (Part α) where one := pure 1


@[to_additive]
instance [Mul α] : Mul (Part α) where mul a b := (· * ·) <$> a <*> b


@[to_additive]
instance [Inv α] : Inv (Part α) where inv := map Inv.inv


@[to_additive]
instance [Div α] : Div (Part α) where div a b := (· / ·) <$> a <*> b


instance [Mod α] : Mod (Part α) where mod a b := (· % ·) <$> a <*> b


instance [Append α] : Append (Part α) where append a b := (· ++ ·) <$> a <*> b


instance [Inter α] : Inter (Part α) where inter a b := (· ∩ ·) <$> a <*> b


instance [Union α] : Union (Part α) where union a b := (· ∪ ·) <$> a <*> b


instance [SDiff α] : SDiff (Part α) where sdiff a b := (· \ ·) <$> a <*> b


theorem mul_def [Mul α] (a b : Part α) : a * b = bind a fun y ↦ map (y * ·) b := rfl

theorem one_def [One α] : (1 : Part α) = some 1 := rfl

theorem inv_def [Inv α] (a : Part α) : a⁻¹ = Part.map (· ⁻¹) a := rfl

theorem div_def [Div α] (a b : Part α) : a / b = bind a fun y => map (y / ·) b := rfl

theorem mod_def [Mod α] (a b : Part α) : a % b = bind a fun y => map (y % ·) b := rfl

theorem append_def [Append α] (a b : Part α) : a ++ b = bind a fun y => map (y ++ ·) b := rfl

theorem inter_def [Inter α] (a b : Part α) : a ∩ b = bind a fun y => map (y ∩ ·) b := rfl

theorem union_def [Union α] (a b : Part α) : a ∪ b = bind a fun y => map (y ∪ ·) b := rfl

theorem sdiff_def [SDiff α] (a b : Part α) : a \ b = bind a fun y => map (y \ ·) b := rfl


@[to_additive]
theorem one_mem_one [One α] : (1 : α) ∈ (1 : Part α) :=
  ⟨trivial, rfl⟩


@[to_additive]
theorem mul_mem_mul [Mul α] (a b : Part α) (ma mb : α) (ha : ma ∈ a) (hb : mb ∈ b) :
                                         /-
                                           α : Type u_1
                                           inst✝ : Mul α
                                           a b : Part α
                                           ma mb : α
                                           ha : Membership.mem a ma
                                           hb : Membership.mem b mb
                                           ⊢ Eq ((HMul.hMul a b).get ⋯) (HMul.hMul ma mb)
                                         -/
    ma * mb ∈ a * b := ⟨⟨ha.1, hb.1⟩, by simp only [← ha.2, ← hb.2]; rfl⟩
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[to_additive]
theorem left_dom_of_mul_dom [Mul α] {a b : Part α} (hab : Dom (a * b)) : a.Dom := hab.1


@[to_additive]
theorem right_dom_of_mul_dom [Mul α] {a b : Part α} (hab : Dom (a * b)) : b.Dom := hab.2


@[to_additive (attr := simp)]
theorem mul_get_eq [Mul α] (a b : Part α) (hab : Dom (a * b)) :
    (a * b).get hab = a.get (left_dom_of_mul_dom hab) * b.get (right_dom_of_mul_dom hab) := rfl


@[to_additive]
                                                                               /-
                                                                                 α : Type u_1
                                                                                 inst✝ : Mul α
                                                                                 a b : α
                                                                                 ⊢ Eq (HMul.hMul (Part.some a) (Part.some b)) (Part.some (HMul.hMul a b))
                                                                               -/
theorem some_mul_some [Mul α] (a b : α) : some a * some b = some (a * b) := by simp [mul_def]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


@[to_additive]
theorem inv_mem_inv [Inv α] (a : Part α) (ma : α) (ha : ma ∈ a) : ma⁻¹ ∈ a⁻¹ := by
  /-
    α : Type u_1
    inst✝ : Inv α
    a : Part α
    ma : α
    ha : Membership.mem a ma
    ⊢ Membership.mem (Inv.inv a) (Inv.inv ma)
  -/
  simp [inv_def]; aesop
                  /-
                    🎉 no goals
                  -/


@[to_additive]
theorem inv_some [Inv α] (a : α) : (some a)⁻¹ = some a⁻¹ :=
  rfl


@[to_additive]
theorem div_mem_div [Div α] (a b : Part α) (ma mb : α) (ha : ma ∈ a) (hb : mb ∈ b) :
                          /-
                            α : Type u_1
                            inst✝ : Div α
                            a b : Part α
                            ma mb : α
                            ha : Membership.mem a ma
                            hb : Membership.mem b mb
                            ⊢ Membership.mem (HDiv.hDiv a b) (HDiv.hDiv ma mb)
                          -/
    ma / mb ∈ a / b := by simp [div_def]; aesop
                                          /-
                                            🎉 no goals
                                          -/


@[to_additive]
theorem left_dom_of_div_dom [Div α] {a b : Part α} (hab : Dom (a / b)) : a.Dom := hab.1


@[to_additive]
theorem right_dom_of_div_dom [Div α] {a b : Part α} (hab : Dom (a / b)) : b.Dom := hab.2


@[to_additive (attr := simp)]
theorem div_get_eq [Div α] (a b : Part α) (hab : Dom (a / b)) :
    (a / b).get hab = a.get (left_dom_of_div_dom hab) / b.get (right_dom_of_div_dom hab) := by
  /-
    α : Type u_1
    inst✝ : Div α
    a b : Part α
    hab : (HDiv.hDiv a b).Dom
    ⊢ Eq ((HDiv.hDiv a b).get hab) (HDiv.hDiv (a.get ⋯) (b.get ⋯))
  -/
  simp [div_def]; aesop
                  /-
                    🎉 no goals
                  -/


@[to_additive]
                                                                               /-
                                                                                 α : Type u_1
                                                                                 inst✝ : Div α
                                                                                 a b : α
                                                                                 ⊢ Eq (HDiv.hDiv (Part.some a) (Part.some b)) (Part.some (HDiv.hDiv a b))
                                                                               -/
theorem some_div_some [Div α] (a b : α) : some a / some b = some (a / b) := by simp [div_def]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


theorem mod_mem_mod [Mod α] (a b : Part α) (ma mb : α) (ha : ma ∈ a) (hb : mb ∈ b) :
                          /-
                            α : Type u_1
                            inst✝ : Mod α
                            a b : Part α
                            ma mb : α
                            ha : Membership.mem a ma
                            hb : Membership.mem b mb
                            ⊢ Membership.mem (HMod.hMod a b) (HMod.hMod ma mb)
                          -/
    ma % mb ∈ a % b := by simp [mod_def]; aesop
                                          /-
                                            🎉 no goals
                                          -/


theorem left_dom_of_mod_dom [Mod α] {a b : Part α} (hab : Dom (a % b)) : a.Dom := hab.1


theorem right_dom_of_mod_dom [Mod α] {a b : Part α} (hab : Dom (a % b)) : b.Dom := hab.2


@[simp]
theorem mod_get_eq [Mod α] (a b : Part α) (hab : Dom (a % b)) :
    (a % b).get hab = a.get (left_dom_of_mod_dom hab) % b.get (right_dom_of_mod_dom hab) := by
  /-
    α : Type u_1
    inst✝ : Mod α
    a b : Part α
    hab : (HMod.hMod a b).Dom
    ⊢ Eq ((HMod.hMod a b).get hab) (HMod.hMod (a.get ⋯) (b.get ⋯))
  -/
  simp [mod_def]; aesop
                  /-
                    🎉 no goals
                  -/


                                                                               /-
                                                                                 α : Type u_1
                                                                                 inst✝ : Mod α
                                                                                 a b : α
                                                                                 ⊢ Eq (HMod.hMod (Part.some a) (Part.some b)) (Part.some (HMod.hMod a b))
                                                                               -/
theorem some_mod_some [Mod α] (a b : α) : some a % some b = some (a % b) := by simp [mod_def]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


theorem append_mem_append [Append α] (a b : Part α) (ma mb : α) (ha : ma ∈ a) (hb : mb ∈ b) :
                            /-
                              α : Type u_1
                              inst✝ : Append α
                              a b : Part α
                              ma mb : α
                              ha : Membership.mem a ma
                              hb : Membership.mem b mb
                              ⊢ Membership.mem (HAppend.hAppend a b) (HAppend.hAppend ma mb)
                            -/
    ma ++ mb ∈ a ++ b := by simp [append_def]; aesop
                                               /-
                                                 🎉 no goals
                                               -/


theorem left_dom_of_append_dom [Append α] {a b : Part α} (hab : Dom (a ++ b)) : a.Dom := hab.1


theorem right_dom_of_append_dom [Append α] {a b : Part α} (hab : Dom (a ++ b)) : b.Dom := hab.2


@[simp]
theorem append_get_eq [Append α] (a b : Part α) (hab : Dom (a ++ b)) : (a ++ b).get hab =
    a.get (left_dom_of_append_dom hab) ++ b.get (right_dom_of_append_dom hab) := by
  /-
    α : Type u_1
    inst✝ : Append α
    a b : Part α
    hab : (HAppend.hAppend a b).Dom
    ⊢ Eq ((HAppend.hAppend a b).get hab) (HAppend.hAppend (a.get ⋯) (b.get ⋯))
  -/
  simp [append_def]; aesop
                     /-
                       🎉 no goals
                     -/


theorem some_append_some [Append α] (a b : α) : some a ++ some b = some (a ++ b) := by
  /-
    α : Type u_1
    inst✝ : Append α
    a b : α
    ⊢ Eq (HAppend.hAppend (Part.some a) (Part.some b)) (Part.some (HAppend.hAppend …
  -/
  simp [append_def]
  /-
    🎉 no goals
  -/


theorem inter_mem_inter [Inter α] (a b : Part α) (ma mb : α) (ha : ma ∈ a) (hb : mb ∈ b) :
                          /-
                            α : Type u_1
                            inst✝ : Inter α
                            a b : Part α
                            ma mb : α
                            ha : Membership.mem a ma
                            hb : Membership.mem b mb
                            ⊢ Membership.mem (Inter.inter a b) (Inter.inter ma mb)
                          -/
    ma ∩ mb ∈ a ∩ b := by simp [inter_def]; aesop
                                            /-
                                              🎉 no goals
                                            -/


theorem left_dom_of_inter_dom [Inter α] {a b : Part α} (hab : Dom (a ∩ b)) : a.Dom := hab.1


theorem right_dom_of_inter_dom [Inter α] {a b : Part α} (hab : Dom (a ∩ b)) : b.Dom := hab.2


@[simp]
theorem inter_get_eq [Inter α] (a b : Part α) (hab : Dom (a ∩ b)) :
    (a ∩ b).get hab = a.get (left_dom_of_inter_dom hab) ∩ b.get (right_dom_of_inter_dom hab) := by
  /-
    α : Type u_1
    inst✝ : Inter α
    a b : Part α
    hab : (Inter.inter a b).Dom
    ⊢ Eq ((Inter.inter a b).get hab) (Inter.inter (a.get ⋯) (b.get ⋯))
  -/
  simp [inter_def]; aesop
                    /-
                      🎉 no goals
                    -/


theorem some_inter_some [Inter α] (a b : α) : some a ∩ some b = some (a ∩ b) := by
  /-
    α : Type u_1
    inst✝ : Inter α
    a b : α
    ⊢ Eq (Inter.inter (Part.some a) (Part.some b)) (Part.some (Inter.inter a b))
  -/
  simp [inter_def]
  /-
    🎉 no goals
  -/


theorem union_mem_union [Union α] (a b : Part α) (ma mb : α) (ha : ma ∈ a) (hb : mb ∈ b) :
                          /-
                            α : Type u_1
                            inst✝ : Union α
                            a b : Part α
                            ma mb : α
                            ha : Membership.mem a ma
                            hb : Membership.mem b mb
                            ⊢ Membership.mem (Union.union a b) (Union.union ma mb)
                          -/
    ma ∪ mb ∈ a ∪ b := by simp [union_def]; aesop
                                            /-
                                              🎉 no goals
                                            -/


theorem left_dom_of_union_dom [Union α] {a b : Part α} (hab : Dom (a ∪ b)) : a.Dom := hab.1


theorem right_dom_of_union_dom [Union α] {a b : Part α} (hab : Dom (a ∪ b)) : b.Dom := hab.2


@[simp]
theorem union_get_eq [Union α] (a b : Part α) (hab : Dom (a ∪ b)) :
    (a ∪ b).get hab = a.get (left_dom_of_union_dom hab) ∪ b.get (right_dom_of_union_dom hab) := by
  /-
    α : Type u_1
    inst✝ : Union α
    a b : Part α
    hab : (Union.union a b).Dom
    ⊢ Eq ((Union.union a b).get hab) (Union.union (a.get ⋯) (b.get ⋯))
  -/
  simp [union_def]; aesop
                    /-
                      🎉 no goals
                    -/


                                                                                   /-
                                                                                     α : Type u_1
                                                                                     inst✝ : Union α
                                                                                     a b : α
                                                                                     ⊢ Eq (Union.union (Part.some a) (Part.some b)) (Part.some (Union.union a b))
                                                                                   -/
theorem some_union_some [Union α] (a b : α) : some a ∪ some b = some (a ∪ b) := by simp [union_def]
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


theorem sdiff_mem_sdiff [SDiff α] (a b : Part α) (ma mb : α) (ha : ma ∈ a) (hb : mb ∈ b) :
                          /-
                            α : Type u_1
                            inst✝ : SDiff α
                            a b : Part α
                            ma mb : α
                            ha : Membership.mem a ma
                            hb : Membership.mem b mb
                            ⊢ Membership.mem (SDiff.sdiff a b) (SDiff.sdiff ma mb)
                          -/
    ma \ mb ∈ a \ b := by simp [sdiff_def]; aesop
                                            /-
                                              🎉 no goals
                                            -/


theorem left_dom_of_sdiff_dom [SDiff α] {a b : Part α} (hab : Dom (a \ b)) : a.Dom := hab.1


theorem right_dom_of_sdiff_dom [SDiff α] {a b : Part α} (hab : Dom (a \ b)) : b.Dom := hab.2


@[simp]
theorem sdiff_get_eq [SDiff α] (a b : Part α) (hab : Dom (a \ b)) :
    (a \ b).get hab = a.get (left_dom_of_sdiff_dom hab) \ b.get (right_dom_of_sdiff_dom hab) := by
  /-
    α : Type u_1
    inst✝ : SDiff α
    a b : Part α
    hab : (SDiff.sdiff a b).Dom
    ⊢ Eq ((SDiff.sdiff a b).get hab) (SDiff.sdiff (a.get ⋯) (b.get ⋯))
  -/
  simp [sdiff_def]; aesop
                    /-
                      🎉 no goals
                    -/


                                                                                   /-
                                                                                     α : Type u_1
                                                                                     inst✝ : SDiff α
                                                                                     a b : α
                                                                                     ⊢ Eq (SDiff.sdiff (Part.some a) (Part.some b)) (Part.some (SDiff.sdiff a b))
                                                                                   -/
theorem some_sdiff_some [SDiff α] (a b : α) : some a \ some b = some (a \ b) := by simp [sdiff_def]
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


