/-- Cofinality of a reflexive order `≼`. This is the smallest cardinality
  of a subset `S : Set α` such that `∀ a, ∃ b ∈ S, a ≼ b`. -/
def cof (r : α → α → Prop) : Cardinal :=
  sInf { c | ∃ S : Set α, (∀ a, ∃ b ∈ S, r a b) ∧ #S = c }


/-- The set in the definition of `Order.cof` is nonempty. -/
private theorem cof_nonempty (r : α → α → Prop) [IsRefl α r] :
    { c | ∃ S : Set α, (∀ a, ∃ b ∈ S, r a b) ∧ #S = c }.Nonempty :=
  ⟨_, Set.univ, fun a => ⟨a, ⟨⟩, refl _⟩, rfl⟩


theorem cof_le (r : α → α → Prop) {S : Set α} (h : ∀ a, ∃ b ∈ S, r a b) : cof r ≤ #S :=
  csInf_le' ⟨S, h, rfl⟩


theorem le_cof [IsRefl α r] (c : Cardinal) :
    c ≤ cof r ↔ ∀ {S : Set α}, (∀ a, ∃ b ∈ S, r a b) → c ≤ #S := by
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : IsRefl α r
    c : Cardinal.{u}
    ⊢ Iff (LE.le c (Order.cof r)) (∀ {S : Set α}, (∀ (a : α), Exists fun b => And  …
  -/
  rw [cof, le_csInf_iff'' (cof_nonempty r)]
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : IsRefl α r
    c : Cardinal.{u}
    ⊢ Iff (∀ (b : Cardinal.{u}), Membership.mem (setOf fun c => Exists fun S => An …
  -/
  use fun H S h => H _ ⟨S, h, rfl⟩
  /-
    case mpr
    α : Type u
    r : α → α → Prop
    inst✝ : IsRefl α r
    c : Cardinal.{u}
    ⊢ (∀ {S : Set α}, (∀ (a : α), Exists fun b => And (Membership.mem S b) (r a b) …
  -/
  rintro H d ⟨S, h, rfl⟩
  /-
    case mpr.intro.intro
    α : Type u
    r : α → α → Prop
    inst✝ : IsRefl α r
    c : Cardinal.{u}
    H : ∀ {S : Set α}, (∀ (a : α), Exists fun b => And (Membership.mem S b) (r a b …
    S : Set α
    h : ∀ (a : α), Exists fun b => And (Membership.mem S b) (r a b)
    ⊢ LE.le c (Cardinal.mk ↑S)
  -/
  exact H h
  /-
    🎉 no goals
  -/


private theorem cof_le_lift [IsRefl β s] (f : r ≃r s) :
    Cardinal.lift.{v} (Order.cof r) ≤ Cardinal.lift.{u} (Order.cof s) := by
  /-
    α : Type u
    β : Type v
    r : α → α → Prop
    s : β → β → Prop
    inst✝ : IsRefl β s
    f : RelIso r s
    ⊢ LE.le (Cardinal.lift.{v, u} (Order.cof r)) (Cardinal.lift.{u, v} (Order.cof  …
  -/
  rw [Order.cof, Order.cof, lift_sInf, lift_sInf, le_csInf_iff'' ((Order.cof_nonempty s).image _)]
  /-
    α : Type u
    β : Type v
    r : α → α → Prop
    s : β → β → Prop
    inst✝ : IsRefl β s
    f : RelIso r s
    ⊢ ∀ (b : Cardinal.{max u v}), Membership.mem (Set.image Cardinal.lift.{u, v} ( …
  -/
  rintro - ⟨-, ⟨u, H, rfl⟩, rfl⟩
  /-
    case intro.intro.intro.intro
    α : Type u
    β : Type v
    r : α → α → Prop
    s : β → β → Prop
    inst✝ : IsRefl β s
    f : RelIso r s
    u : Set β
    H : ∀ (a : β), Exists fun b => And (Membership.mem u b) (s a b)
    ⊢ LE.le (InfSet.sInf (Set.image Cardinal.lift.{v, u} (setOf fun c => Exists fu …
  -/
  apply csInf_le'
  /-
    case intro.intro.intro.intro.h
    α : Type u
    β : Type v
    r : α → α → Prop
    s : β → β → Prop
    inst✝ : IsRefl β s
    f : RelIso r s
    u : Set β
    H : ∀ (a : β), Exists fun b => And (Membership.mem u b) (s a b)
    ⊢ Membership.mem (Set.image Cardinal.lift.{v, u} (setOf fun c => Exists fun S  …
  -/
  refine ⟨_, ⟨f.symm '' u, fun a => ?_, rfl⟩, lift_mk_eq'.2 ⟨(f.symm.toEquiv.image u).symm⟩⟩
  /-
    case intro.intro.intro.intro.h
    α : Type u
    β : Type v
    r : α → α → Prop
    s : β → β → Prop
    inst✝ : IsRefl β s
    f : RelIso r s
    u : Set β
    H : ∀ (a : β), Exists fun b => And (Membership.mem u b) (s a b)
    a : α
    ⊢ Exists fun b => And (Membership.mem (Set.image (⇑f.symm) u) b) (r a b)
  -/
  rcases H (f a) with ⟨b, hb, hb'⟩
  /-
    case intro.intro.intro.intro.h.intro.intro
    α : Type u
    β : Type v
    r : α → α → Prop
    s : β → β → Prop
    inst✝ : IsRefl β s
    f : RelIso r s
    u : Set β
    H : ∀ (a : β), Exists fun b => And (Membership.mem u b) (s a b)
    a : α
    b : β
    hb : Membership.mem u b
    hb' : s (f a) b
    ⊢ Exists fun b => And (Membership.mem (Set.image (⇑f.symm) u) b) (r a b)
  -/
  refine ⟨f.symm b, mem_image_of_mem _ hb, f.map_rel_iff.1 ?_⟩
  /-
    case intro.intro.intro.intro.h.intro.intro
    α : Type u
    β : Type v
    r : α → α → Prop
    s : β → β → Prop
    inst✝ : IsRefl β s
    f : RelIso r s
    u : Set β
    H : ∀ (a : β), Exists fun b => And (Membership.mem u b) (s a b)
    a : α
    b : β
    hb : Membership.mem u b
    hb' : s (f a) b
    ⊢ s (f a) (f (f.symm b))
  -/
  rwa [RelIso.apply_symm_apply]
  /-
    🎉 no goals
  -/


theorem cof_eq_lift [IsRefl β s] (f : r ≃r s) :
    Cardinal.lift.{v} (Order.cof r) = Cardinal.lift.{u} (Order.cof s) :=
  have := f.toRelEmbedding.isRefl
  (f.cof_le_lift).antisymm (f.symm.cof_le_lift)


theorem cof_eq {α β : Type u} {r : α → α → Prop} {s} [IsRefl β s] (f : r ≃r s) :
    Order.cof r = Order.cof s :=
  lift_inj.1 (f.cof_eq_lift)


@[deprecated cof_eq (since := "2024-10-22")]
theorem cof_le {α β : Type u} {r : α → α → Prop} {s} [IsRefl β s] (f : r ≃r s) :
    Order.cof r ≤ Order.cof s :=
  f.cof_eq.le


/-- Cofinality of a strict order `≺`. This is the smallest cardinality of a set `S : Set α` such
that `∀ a, ∃ b ∈ S, ¬ b ≺ a`. -/
@[deprecated Order.cof (since := "2024-10-22")]
def StrictOrder.cof (r : α → α → Prop) : Cardinal :=
  Order.cof (swap rᶜ)


/-- The set in the definition of `Order.StrictOrder.cof` is nonempty. -/
@[deprecated "No deprecation message was provided." (since := "2024-10-22")]
theorem StrictOrder.cof_nonempty (r : α → α → Prop) [IsIrrefl α r] :
    { c | ∃ S : Set α, Unbounded r S ∧ #S = c }.Nonempty :=
  @Order.cof_nonempty α _ (IsRefl.swap rᶜ)


/-- Cofinality of an ordinal. This is the smallest cardinal of a subset `S` of the ordinal which is
unbounded, in the sense `∀ a, ∃ b ∈ S, a ≤ b`.

In particular, `cof 0 = 0` and `cof (succ o) = 1`. -/
def cof (o : Ordinal.{u}) : Cardinal.{u} :=
  o.liftOn (fun a ↦ Order.cof (swap a.rᶜ)) fun _ _ ⟨f⟩ ↦ f.compl.swap.cof_eq


theorem cof_type (r : α → α → Prop) [IsWellOrder α r] : (type r).cof = Order.cof (swap rᶜ) :=
  rfl


theorem cof_type_lt [LinearOrder α] [IsWellOrder α (· < ·)] :
    (@type α (· < ·) _).cof = @Order.cof α (· ≤ ·) := by
  /-
    α : Type u
    inst✝¹ : LinearOrder α
    inst✝ : IsWellOrder α fun x1 x2 => LT.lt x1 x2
    ⊢ Eq (Ordinal.type fun x1 x2 => LT.lt x1 x2).cof (Order.cof fun x1 x2 => LE.le …
  -/
  rw [cof_type, compl_lt, swap_ge]
  /-
    🎉 no goals
  -/


theorem cof_eq_cof_toType (o : Ordinal) : o.cof = @Order.cof o.toType (· ≤ ·) := by
  /-
    o : Ordinal.{u_1}
    ⊢ Eq o.cof (Order.cof fun x1 x2 => LE.le x1 x2)
  -/
  conv_lhs => rw [← type_toType o, cof_type_lt]
  /-
    🎉 no goals
  -/


theorem le_cof_type [IsWellOrder α r] {c} : c ≤ cof (type r) ↔ ∀ S, Unbounded r S → c ≤ #S :=
  (le_csInf_iff'' (Order.cof_nonempty _)).trans
    ⟨fun H S h => H _ ⟨S, h, rfl⟩, by
      /-
        α : Type u
        r : α → α → Prop
        inst✝ : IsWellOrder α r
        c : Cardinal.{u}
        ⊢ (∀ (S : Set α), Set.Unbounded r S → LE.le c (Cardinal.mk ↑S)) → ∀ (b : Cardi …
      -/
      rintro H d ⟨S, h, rfl⟩
      /-
        case intro.intro
        α : Type u
        r : α → α → Prop
        inst✝ : IsWellOrder α r
        c : Cardinal.{u}
        H : ∀ (S : Set α), Set.Unbounded r S → LE.le c (Cardinal.mk ↑S)
        S : Set { α := α, r := r, wo := inst✝ }.α
        h : ∀ (a : { α := α, r := r, wo := inst✝ }.α), Exists fun b => And (Membership …
        ⊢ LE.le c (Cardinal.mk ↑S)
      -/
      exact H _ h⟩
      /-
        🎉 no goals
      -/


theorem cof_type_le [IsWellOrder α r] {S : Set α} (h : Unbounded r S) : cof (type r) ≤ #S :=
  le_cof_type.1 le_rfl S h


theorem lt_cof_type [IsWellOrder α r] {S : Set α} : #S < cof (type r) → Bounded r S := by
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    S : Set α
    ⊢ LT.lt (Cardinal.mk ↑S) (Ordinal.type r).cof → Set.Bounded r S
  -/
  simpa using not_imp_not.2 cof_type_le
  /-
    🎉 no goals
  -/


theorem cof_eq (r : α → α → Prop) [IsWellOrder α r] : ∃ S, Unbounded r S ∧ #S = cof (type r) :=
  csInf_mem (Order.cof_nonempty (swap rᶜ))


theorem ord_cof_eq (r : α → α → Prop) [IsWellOrder α r] :
    ∃ S, Unbounded r S ∧ type (Subrel r S) = (cof (type r)).ord := by
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    ⊢ Exists fun S => And (Set.Unbounded r S) (Eq (Ordinal.type (Subrel r S)) (Ord …
  -/
  let ⟨S, hS, e⟩ := cof_eq r
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    S : Set α
    hS : Set.Unbounded r S
    e : Eq (Cardinal.mk ↑S) (Ordinal.type r).cof
    ⊢ Exists fun S => And (Set.Unbounded r S) (Eq (Ordinal.type (Subrel r S)) (Ord …
  -/
  let ⟨s, _, e'⟩ := Cardinal.ord_eq S
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    S : Set α
    hS : Set.Unbounded r S
    e : Eq (Cardinal.mk ↑S) (Ordinal.type r).cof
    s : ↑S → ↑S → Prop
    w✝ : IsWellOrder (↑S) s
    e' : Eq (Cardinal.mk ↑S).ord (Ordinal.type s)
    ⊢ Exists fun S => And (Set.Unbounded r S) (Eq (Ordinal.type (Subrel r S)) (Ord …
  -/
  let T : Set α := { a | ∃ aS : a ∈ S, ∀ b : S, s b ⟨_, aS⟩ → r b a }
  suffices Unbounded r T by
    refine ⟨T, this, le_antisymm ?_ (Cardinal.ord_le.2 <| cof_type_le this)⟩
    rw [← e, e']
    refine
      (RelEmbedding.ofMonotone
          (fun a : T =>
            (⟨a,
                let ⟨aS, _⟩ := a.2
                aS⟩ :
              S))
          fun a b h => ?_).ordinal_type_le
    rcases a with ⟨a, aS, ha⟩
    rcases b with ⟨b, bS, hb⟩
    change s ⟨a, _⟩ ⟨b, _⟩
    refine ((trichotomous_of s _ _).resolve_left fun hn => ?_).resolve_left ?_
    · exact asymm h (ha _ hn)
    · intro e
      injection e with e
      subst b
      exact irrefl _ h
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    S : Set α
    hS : Set.Unbounded r S
    e : Eq (Cardinal.mk ↑S) (Ordinal.type r).cof
    s : ↑S → ↑S → Prop
    w✝ : IsWellOrder (↑S) s
    e' : Eq (Cardinal.mk ↑S).ord (Ordinal.type s)
    T : Set α := setOf fun a => Exists fun aS => ∀ (b : ↑S), s b ⟨a, aS⟩ → r (↑b) a
    ⊢ Set.Unbounded r T
  -/
  intro a
  have : { b : S | ¬r b a }.Nonempty :=
    let ⟨b, bS, ba⟩ := hS a
    ⟨⟨b, bS⟩, ba⟩
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    S : Set α
    hS : Set.Unbounded r S
    e : Eq (Cardinal.mk ↑S) (Ordinal.type r).cof
    s : ↑S → ↑S → Prop
    w✝ : IsWellOrder (↑S) s
    e' : Eq (Cardinal.mk ↑S).ord (Ordinal.type s)
    T : Set α := setOf fun a => Exists fun aS => ∀ (b : ↑S), s b ⟨a, aS⟩ → r (↑b) a
    a : α
    this : (setOf fun b => Not (r (↑b) a)).Nonempty
    ⊢ Exists fun b => And (Membership.mem T b) (Not (r b a))
  -/
  let b := (IsWellFounded.wf : WellFounded s).min _ this
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    S : Set α
    hS : Set.Unbounded r S
    e : Eq (Cardinal.mk ↑S) (Ordinal.type r).cof
    s : ↑S → ↑S → Prop
    w✝ : IsWellOrder (↑S) s
    e' : Eq (Cardinal.mk ↑S).ord (Ordinal.type s)
    T : Set α := setOf fun a => Exists fun aS => ∀ (b : ↑S), s b ⟨a, aS⟩ → r (↑b) a
    a : α
    this : (setOf fun b => Not (r (↑b) a)).Nonempty
    b : ↑S := ⋯.min (setOf fun b => Not (r (↑b) a)) this
    ⊢ Exists fun b => And (Membership.mem T b) (Not (r b a))
  -/
  have ba : ¬r b a := IsWellFounded.wf.min_mem _ this
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    S : Set α
    hS : Set.Unbounded r S
    e : Eq (Cardinal.mk ↑S) (Ordinal.type r).cof
    s : ↑S → ↑S → Prop
    w✝ : IsWellOrder (↑S) s
    e' : Eq (Cardinal.mk ↑S).ord (Ordinal.type s)
    T : Set α := setOf fun a => Exists fun aS => ∀ (b : ↑S), s b ⟨a, aS⟩ → r (↑b) a
    a : α
    this : (setOf fun b => Not (r (↑b) a)).Nonempty
    b : ↑S := ⋯.min (setOf fun b => Not (r (↑b) a)) this
    ba : Not (r (↑b) a)
    ⊢ Exists fun b => And (Membership.mem T b) (Not (r b a))
  -/
  refine ⟨b, ⟨b.2, fun c => not_imp_not.1 fun h => ?_⟩, ba⟩
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    S : Set α
    hS : Set.Unbounded r S
    e : Eq (Cardinal.mk ↑S) (Ordinal.type r).cof
    s : ↑S → ↑S → Prop
    w✝ : IsWellOrder (↑S) s
    e' : Eq (Cardinal.mk ↑S).ord (Ordinal.type s)
    T : Set α := setOf fun a => Exists fun aS => ∀ (b : ↑S), s b ⟨a, aS⟩ → r (↑b) a
    a : α
    this : (setOf fun b => Not (r (↑b) a)).Nonempty
    b : ↑S := ⋯.min (setOf fun b => Not (r (↑b) a)) this
    ba : Not (r (↑b) a)
    c : ↑S
    h : Not (r ↑c ↑b)
    ⊢ Not (s c ⟨↑b, ⋯⟩)
  -/
  rw [show ∀ b : S, (⟨b, b.2⟩ : S) = b by intro b; cases b; rfl]
  /-
    α : Type u
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    S : Set α
    hS : Set.Unbounded r S
    e : Eq (Cardinal.mk ↑S) (Ordinal.type r).cof
    s : ↑S → ↑S → Prop
    w✝ : IsWellOrder (↑S) s
    e' : Eq (Cardinal.mk ↑S).ord (Ordinal.type s)
    T : Set α := setOf fun a => Exists fun aS => ∀ (b : ↑S), s b ⟨a, aS⟩ → r (↑b) a
    a : α
    this : (setOf fun b => Not (r (↑b) a)).Nonempty
    b : ↑S := ⋯.min (setOf fun b => Not (r (↑b) a)) this
    ba : Not (r (↑b) a)
    c : ↑S
    h : Not (r ↑c ↑b)
    ⊢ Not (s c b)
  -/
  exact IsWellFounded.wf.not_lt_min _ this (IsOrderConnected.neg_trans h ba)
  /-
    🎉 no goals
  -/


private theorem card_mem_cof {o} : ∃ (ι : _) (f : ι → Ordinal), lsub.{u, u} f = o ∧ #ι = o.card :=
  ⟨_, _, lsub_typein o, mk_toType o⟩


/-- The set in the `lsub` characterization of `cof` is nonempty. -/
theorem cof_lsub_def_nonempty (o) :
    { a : Cardinal | ∃ (ι : _) (f : ι → Ordinal), lsub.{u, u} f = o ∧ #ι = a }.Nonempty :=
  ⟨_, card_mem_cof⟩


theorem cof_eq_sInf_lsub (o : Ordinal.{u}) : cof o =
    sInf { a : Cardinal | ∃ (ι : Type u) (f : ι → Ordinal), lsub.{u, u} f = o ∧ #ι = a } := by
  /-
    o : Ordinal.{u}
    ⊢ Eq o.cof (InfSet.sInf (setOf fun a => Exists fun ι => Exists fun f => And (E …
  -/
  refine le_antisymm (le_csInf (cof_lsub_def_nonempty o) ?_) (csInf_le' ?_)
    /-
      case refine_1
      o : Ordinal.{u}
      ⊢ ∀ (b : Cardinal.{u}), Membership.mem (setOf fun a => Exists fun ι => Exists  …
    -/
  · rintro a ⟨ι, f, hf, rfl⟩
    /-
      case refine_1.intro.intro.intro
      o : Ordinal.{u}
      ι : Type u
      f : ι → Ordinal.{u}
      hf : Eq (Ordinal.lsub f) o
      ⊢ LE.le o.cof (Cardinal.mk ι)
    -/
    rw [← type_toType o]
    refine
      (cof_type_le fun a => ?_).trans
        (@mk_le_of_injective _ _
          (fun s : typein ((· < ·) : o.toType → o.toType → Prop) ⁻¹' Set.range f =>
            Classical.choose s.prop)
          fun s t hst => by
          let H := congr_arg f hst
          rwa [Classical.choose_spec s.prop, Classical.choose_spec t.prop, typein_inj,
            Subtype.coe_inj] at H)
    /-
      case refine_1.intro.intro.intro
      o : Ordinal.{u}
      ι : Type u
      f : ι → Ordinal.{u}
      hf : Eq (Ordinal.lsub f) o
      a : o.toType
      ⊢ Exists fun b => And (Membership.mem (Set.preimage (⇑(Ordinal.typein fun x1 x …
    -/
    have := typein_lt_self a
    /-
      case refine_1.intro.intro.intro
      o : Ordinal.{u}
      ι : Type u
      f : ι → Ordinal.{u}
      hf : Eq (Ordinal.lsub f) o
      a : o.toType
      this : LT.lt ((Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding a) o
      ⊢ Exists fun b => And (Membership.mem (Set.preimage (⇑(Ordinal.typein fun x1 x …
    -/
    simp_rw [← hf, lt_lsub_iff] at this
    /-
      case refine_1.intro.intro.intro
      o : Ordinal.{u}
      ι : Type u
      f : ι → Ordinal.{u}
      hf : Eq (Ordinal.lsub f) o
      a : o.toType
      this : Exists fun i => LE.le ((Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelE …
      ⊢ Exists fun b => And (Membership.mem (Set.preimage (⇑(Ordinal.typein fun x1 x …
    -/
    cases' this with i hi
    /-
      case refine_1.intro.intro.intro.intro
      o : Ordinal.{u}
      ι : Type u
      f : ι → Ordinal.{u}
      hf : Eq (Ordinal.lsub f) o
      a : o.toType
      i : ι
      hi : LE.le ((Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding a) (f i)
      ⊢ Exists fun b => And (Membership.mem (Set.preimage (⇑(Ordinal.typein fun x1 x …
    -/
    refine ⟨enum (α := o.toType) (· < ·) ⟨f i, ?_⟩, ?_, ?_⟩
      /-
        case refine_1.intro.intro.intro.intro.refine_1
        o : Ordinal.{u}
        ι : Type u
        f : ι → Ordinal.{u}
        hf : Eq (Ordinal.lsub f) o
        a : o.toType
        i : ι
        hi : LE.le ((Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding a) (f i)
        ⊢ LT.lt (f i) (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      -/
    · rw [type_toType, ← hf]
      /-
        case refine_1.intro.intro.intro.intro.refine_1
        o : Ordinal.{u}
        ι : Type u
        f : ι → Ordinal.{u}
        hf : Eq (Ordinal.lsub f) o
        a : o.toType
        i : ι
        hi : LE.le ((Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding a) (f i)
        ⊢ LT.lt (f i) (Ordinal.lsub f)
      -/
      apply lt_lsub
      /-
        🎉 no goals
      -/
      /-
        case refine_1.intro.intro.intro.intro.refine_2
        o : Ordinal.{u}
        ι : Type u
        f : ι → Ordinal.{u}
        hf : Eq (Ordinal.lsub f) o
        a : o.toType
        i : ι
        hi : LE.le ((Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding a) (f i)
        ⊢ Membership.mem (Set.preimage (⇑(Ordinal.typein fun x1 x2 => LT.lt x1 x2).toR …
      -/
    · rw [mem_preimage, typein_enum]
      /-
        case refine_1.intro.intro.intro.intro.refine_2
        o : Ordinal.{u}
        ι : Type u
        f : ι → Ordinal.{u}
        hf : Eq (Ordinal.lsub f) o
        a : o.toType
        i : ι
        hi : LE.le ((Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding a) (f i)
        ⊢ Membership.mem (Set.range f) (f i)
      -/
      exact mem_range_self i
      /-
        🎉 no goals
      -/
      /-
        case refine_1.intro.intro.intro.intro.refine_3
        o : Ordinal.{u}
        ι : Type u
        f : ι → Ordinal.{u}
        hf : Eq (Ordinal.lsub f) o
        a : o.toType
        i : ι
        hi : LE.le ((Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding a) (f i)
        ⊢ Not (LT.lt ((Ordinal.enum fun x1 x2 => LT.lt x1 x2) ⟨f i, ⋯⟩) a)
      -/
    · rwa [← typein_le_typein, typein_enum]
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      o : Ordinal.{u}
      ⊢ Membership.mem (setOf fun a => Exists fun ι => Exists fun f => And (Eq (Ordi …
    -/
  · rcases cof_eq (α := o.toType) (· < ·) with ⟨S, hS, hS'⟩
    /-
      case refine_2.intro.intro
      o : Ordinal.{u}
      S : Set o.toType
      hS : Set.Unbounded (fun x1 x2 => LT.lt x1 x2) S
      hS' : Eq (Cardinal.mk ↑S) (Ordinal.type fun x1 x2 => LT.lt x1 x2).cof
      ⊢ Membership.mem (setOf fun a => Exists fun ι => Exists fun f => And (Eq (Ordi …
    -/
    let f : S → Ordinal := fun s => typein LT.lt s.val
    refine ⟨S, f, le_antisymm (lsub_le fun i => typein_lt_self (o := o) i)
      (le_of_forall_lt fun a ha => ?_), by rwa [type_toType o] at hS'⟩
    /-
      case refine_2.intro.intro
      o : Ordinal.{u}
      S : Set o.toType
      hS : Set.Unbounded (fun x1 x2 => LT.lt x1 x2) S
      hS' : Eq (Cardinal.mk ↑S) (Ordinal.type fun x1 x2 => LT.lt x1 x2).cof
      f : ↑S → Ordinal.{u} := fun s => (Ordinal.typein LT.lt).toRelEmbedding ↑s
      a : Ordinal.{u}
      ha : LT.lt a o
      ⊢ LT.lt a (Ordinal.lsub f)
    -/
    rw [← type_toType o] at ha
    /-
      case refine_2.intro.intro
      o : Ordinal.{u}
      S : Set o.toType
      hS : Set.Unbounded (fun x1 x2 => LT.lt x1 x2) S
      hS' : Eq (Cardinal.mk ↑S) (Ordinal.type fun x1 x2 => LT.lt x1 x2).cof
      f : ↑S → Ordinal.{u} := fun s => (Ordinal.typein LT.lt).toRelEmbedding ↑s
      a : Ordinal.{u}
      ha : LT.lt a (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      ⊢ LT.lt a (Ordinal.lsub f)
    -/
    rcases hS (enum (· < ·) ⟨a, ha⟩) with ⟨b, hb, hb'⟩
    /-
      case refine_2.intro.intro.intro.intro
      o : Ordinal.{u}
      S : Set o.toType
      hS : Set.Unbounded (fun x1 x2 => LT.lt x1 x2) S
      hS' : Eq (Cardinal.mk ↑S) (Ordinal.type fun x1 x2 => LT.lt x1 x2).cof
      f : ↑S → Ordinal.{u} := fun s => (Ordinal.typein LT.lt).toRelEmbedding ↑s
      a : Ordinal.{u}
      ha : LT.lt a (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      b : o.toType
      hb : Membership.mem S b
      hb' : Not ((fun x1 x2 => LT.lt x1 x2) b ((Ordinal.enum fun x1 x2 => LT.lt x1 x …
      ⊢ LT.lt a (Ordinal.lsub f)
    -/
    rw [← typein_le_typein, typein_enum] at hb'
    /-
      case refine_2.intro.intro.intro.intro
      o : Ordinal.{u}
      S : Set o.toType
      hS : Set.Unbounded (fun x1 x2 => LT.lt x1 x2) S
      hS' : Eq (Cardinal.mk ↑S) (Ordinal.type fun x1 x2 => LT.lt x1 x2).cof
      f : ↑S → Ordinal.{u} := fun s => (Ordinal.typein LT.lt).toRelEmbedding ↑s
      a : Ordinal.{u}
      ha : LT.lt a (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      b : o.toType
      hb : Membership.mem S b
      hb' : LE.le a ((Ordinal.typein LT.lt).toRelEmbedding b)
      ⊢ LT.lt a (Ordinal.lsub f)
    -/
    exact hb'.trans_lt (lt_lsub.{u, u} f ⟨b, hb⟩)
    /-
      🎉 no goals
    -/


@[simp]
theorem lift_cof (o) : Cardinal.lift.{u, v} (cof o) = cof (Ordinal.lift.{u, v} o) := by
  /-
    o : Ordinal.{v}
    ⊢ Eq (Cardinal.lift.{u, v} o.cof) (Ordinal.lift.{u, v} o).cof
  -/
  refine inductionOn o fun α r _ ↦ ?_
  rw [← type_uLift, cof_type, cof_type, ← Cardinal.lift_id'.{v, u} (Order.cof _),
    ← Cardinal.lift_umax]
  /-
    o : Ordinal.{v}
    α : Type v
    r : α → α → Prop
    x✝ : IsWellOrder α r
    ⊢ Eq (Cardinal.lift.{max v u, v} (Order.cof (Function.swap (HasCompl.compl r)) …
  -/
  apply RelIso.cof_eq_lift ⟨Equiv.ulift.symm, _⟩
  /-
    o : Ordinal.{v}
    α : Type v
    r : α → α → Prop
    x✝ : IsWellOrder α r
    ⊢ ∀ {a b : α}, Iff (Function.swap (HasCompl.compl (Order.Preimage ULift.down r …
  -/
  simp [swap]
  /-
    🎉 no goals
  -/


theorem cof_le_card (o) : cof o ≤ card o := by
  /-
    o : Ordinal.{u_1}
    ⊢ LE.le o.cof o.card
  -/
  rw [cof_eq_sInf_lsub]
  /-
    o : Ordinal.{u_1}
    ⊢ LE.le (InfSet.sInf (setOf fun a => Exists fun ι => Exists fun f => And (Eq ( …
  -/
  exact csInf_le' card_mem_cof
  /-
    🎉 no goals
  -/


                                                        /-
                                                          c : Cardinal.{u_1}
                                                          ⊢ LE.le c.ord.cof c
                                                        -/
theorem cof_ord_le (c : Cardinal) : c.ord.cof ≤ c := by simpa using cof_le_card c.ord
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem ord_cof_le (o : Ordinal.{u}) : o.cof.ord ≤ o :=
  (ord_le_ord.2 (cof_le_card o)).trans (ord_card_le o)


theorem exists_lsub_cof (o : Ordinal) :
    ∃ (ι : _) (f : ι → Ordinal), lsub.{u, u} f = o ∧ #ι = cof o := by
  /-
    o : Ordinal.{u}
    ⊢ Exists fun ι => Exists fun f => And (Eq (Ordinal.lsub f) o) (Eq (Cardinal.mk …
  -/
  rw [cof_eq_sInf_lsub]
  /-
    o : Ordinal.{u}
    ⊢ Exists fun ι => Exists fun f => And (Eq (Ordinal.lsub f) o) (Eq (Cardinal.mk …
  -/
  exact csInf_mem (cof_lsub_def_nonempty o)
  /-
    🎉 no goals
  -/


theorem cof_lsub_le {ι} (f : ι → Ordinal) : cof (lsub.{u, u} f) ≤ #ι := by
  /-
    ι : Type u
    f : ι → Ordinal.{u}
    ⊢ LE.le (Ordinal.lsub f).cof (Cardinal.mk ι)
  -/
  rw [cof_eq_sInf_lsub]
  /-
    ι : Type u
    f : ι → Ordinal.{u}
    ⊢ LE.le (InfSet.sInf (setOf fun a => Exists fun ι_1 => Exists fun f_1 => And ( …
  -/
  exact csInf_le' ⟨ι, f, rfl, rfl⟩
  /-
    🎉 no goals
  -/


theorem cof_lsub_le_lift {ι} (f : ι → Ordinal) :
    cof (lsub.{u, v} f) ≤ Cardinal.lift.{v, u} #ι := by
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    ⊢ LE.le (Ordinal.lsub f).cof (Cardinal.lift.{v, u} (Cardinal.mk ι))
  -/
  rw [← mk_uLift.{u, v}]
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    ⊢ LE.le (Ordinal.lsub f).cof (Cardinal.mk (ULift.{v, u} ι))
  -/
  convert cof_lsub_le.{max u v} fun i : ULift.{v, u} ι => f i.down
  exact
    lsub_eq_of_range_eq.{u, max u v, max u v}
      (Set.ext fun x => ⟨fun ⟨i, hi⟩ => ⟨ULift.up.{v, u} i, hi⟩, fun ⟨i, hi⟩ => ⟨_, hi⟩⟩)


theorem le_cof_iff_lsub {o : Ordinal} {a : Cardinal} :
    a ≤ cof o ↔ ∀ {ι} (f : ι → Ordinal), lsub.{u, u} f = o → a ≤ #ι := by
  /-
    o : Ordinal.{u}
    a : Cardinal.{u}
    ⊢ Iff (LE.le a o.cof) (∀ {ι : Type u} (f : ι → Ordinal.{u}), Eq (Ordinal.lsub  …
  -/
  rw [cof_eq_sInf_lsub]
  exact
    (le_csInf_iff'' (cof_lsub_def_nonempty o)).trans
      ⟨fun H ι f hf => H _ ⟨ι, f, hf, rfl⟩, fun H b ⟨ι, f, hf, hb⟩ => by
        rw [← hb]
        exact H _ hf⟩


theorem lsub_lt_ord_lift {ι} {f : ι → Ordinal} {c : Ordinal}
    (hι : Cardinal.lift.{v, u} #ι < c.cof)
    (hf : ∀ i, f i < c) : lsub.{u, v} f < c :=
  lt_of_le_of_ne (lsub_le hf) fun h => by
    /-
      ι : Type u
      f : ι → Ordinal.{max u v}
      c : Ordinal.{max u v}
      hι : LT.lt (Cardinal.lift.{v, u} (Cardinal.mk ι)) c.cof
      hf : ∀ (i : ι), LT.lt (f i) c
      h : Eq (Ordinal.lsub f) c
      ⊢ False
    -/
    subst h
    /-
      ι : Type u
      f : ι → Ordinal.{max u v}
      hι : LT.lt (Cardinal.lift.{v, u} (Cardinal.mk ι)) (Ordinal.lsub f).cof
      hf : ∀ (i : ι), LT.lt (f i) (Ordinal.lsub f)
      ⊢ False
    -/
    exact (cof_lsub_le_lift.{u, v} f).not_lt hι
    /-
      🎉 no goals
    -/


theorem lsub_lt_ord {ι} {f : ι → Ordinal} {c : Ordinal} (hι : #ι < c.cof) :
    (∀ i, f i < c) → lsub.{u, u} f < c :=
                       /-
                         ι : Type u
                         f : ι → Ordinal.{u}
                         c : Ordinal.{u}
                         hι : LT.lt (Cardinal.mk ι) c.cof
                         ⊢ LT.lt (Cardinal.lift.{u, u} (Cardinal.mk ι)) c.cof
                       -/
  lsub_lt_ord_lift (by rwa [(#ι).lift_id])
                       /-
                         🎉 no goals
                       -/


theorem cof_iSup_le_lift {ι} {f : ι → Ordinal} (H : ∀ i, f i < iSup f) :
    cof (iSup f) ≤ Cardinal.lift.{v, u} #ι := by
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    H : ∀ (i : ι), LT.lt (f i) (iSup f)
    ⊢ LE.le (iSup f).cof (Cardinal.lift.{v, u} (Cardinal.mk ι))
  -/
  rw [← Ordinal.sup] at *
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    H : ∀ (i : ι), LT.lt (f i) (Ordinal.sup f)
    ⊢ LE.le (Ordinal.sup f).cof (Cardinal.lift.{v, u} (Cardinal.mk ι))
  -/
  rw [← sup_eq_lsub_iff_lt_sup.{u, v}] at H
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    H : Eq (Ordinal.sup f) (Ordinal.lsub f)
    ⊢ LE.le (Ordinal.sup f).cof (Cardinal.lift.{v, u} (Cardinal.mk ι))
  -/
  rw [H]
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    H : Eq (Ordinal.sup f) (Ordinal.lsub f)
    ⊢ LE.le (Ordinal.lsub f).cof (Cardinal.lift.{v, u} (Cardinal.mk ι))
  -/
  exact cof_lsub_le_lift f
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated cof_iSup_le_lift (since := "2024-08-27")]
theorem cof_sup_le_lift {ι} {f : ι → Ordinal} (H : ∀ i, f i < sup.{u, v} f) :
    cof (sup.{u, v} f) ≤ Cardinal.lift.{v, u} #ι := by
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    H : ∀ (i : ι), LT.lt (f i) (Ordinal.sup f)
    ⊢ LE.le (Ordinal.sup f).cof (Cardinal.lift.{v, u} (Cardinal.mk ι))
  -/
  rw [← sup_eq_lsub_iff_lt_sup.{u, v}] at H
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    H : Eq (Ordinal.sup f) (Ordinal.lsub f)
    ⊢ LE.le (Ordinal.sup f).cof (Cardinal.lift.{v, u} (Cardinal.mk ι))
  -/
  rw [H]
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    H : Eq (Ordinal.sup f) (Ordinal.lsub f)
    ⊢ LE.le (Ordinal.lsub f).cof (Cardinal.lift.{v, u} (Cardinal.mk ι))
  -/
  exact cof_lsub_le_lift f
  /-
    🎉 no goals
  -/


theorem cof_iSup_le {ι} {f : ι → Ordinal} (H : ∀ i, f i < iSup f) :
    cof (iSup f) ≤ #ι := by
  /-
    ι : Type u_1
    f : ι → Ordinal.{u_1}
    H : ∀ (i : ι), LT.lt (f i) (iSup f)
    ⊢ LE.le (iSup f).cof (Cardinal.mk ι)
  -/
  rw [← (#ι).lift_id]
  /-
    ι : Type u_1
    f : ι → Ordinal.{u_1}
    H : ∀ (i : ι), LT.lt (f i) (iSup f)
    ⊢ LE.le (iSup f).cof (Cardinal.lift.{u_1, u_1} (Cardinal.mk ι))
  -/
  exact cof_iSup_le_lift H
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated cof_iSup_le (since := "2024-08-27")]
theorem cof_sup_le {ι} {f : ι → Ordinal} (H : ∀ i, f i < sup.{u, u} f) :
    cof (sup.{u, u} f) ≤ #ι := by
  /-
    ι : Type u
    f : ι → Ordinal.{u}
    H : ∀ (i : ι), LT.lt (f i) (Ordinal.sup f)
    ⊢ LE.le (Ordinal.sup f).cof (Cardinal.mk ι)
  -/
  rw [← (#ι).lift_id]
  /-
    ι : Type u
    f : ι → Ordinal.{u}
    H : ∀ (i : ι), LT.lt (f i) (Ordinal.sup f)
    ⊢ LE.le (Ordinal.sup f).cof (Cardinal.lift.{u, u} (Cardinal.mk ι))
  -/
  exact cof_sup_le_lift H
  /-
    🎉 no goals
  -/


theorem iSup_lt_ord_lift {ι} {f : ι → Ordinal} {c : Ordinal} (hι : Cardinal.lift.{v, u} #ι < c.cof)
    (hf : ∀ i, f i < c) : iSup f < c :=
  (sup_le_lsub.{u, v} f).trans_lt (lsub_lt_ord_lift hι hf)


set_option linter.deprecated false in
@[deprecated iSup_lt_ord_lift (since := "2024-08-27")]
theorem sup_lt_ord_lift {ι} {f : ι → Ordinal} {c : Ordinal} (hι : Cardinal.lift.{v, u} #ι < c.cof)
    (hf : ∀ i, f i < c) : sup.{u, v} f < c :=
  iSup_lt_ord_lift hι hf


theorem iSup_lt_ord {ι} {f : ι → Ordinal} {c : Ordinal} (hι : #ι < c.cof) :
    (∀ i, f i < c) → iSup f < c :=
                       /-
                         ι : Type u_1
                         f : ι → Ordinal.{u_1}
                         c : Ordinal.{u_1}
                         hι : LT.lt (Cardinal.mk ι) c.cof
                         ⊢ LT.lt (Cardinal.lift.{?u.29217, u_1} (Cardinal.mk ι)) c.cof
                       -/
  iSup_lt_ord_lift (by rwa [(#ι).lift_id])
                       /-
                         🎉 no goals
                       -/


set_option linter.deprecated false in
@[deprecated iSup_lt_ord (since := "2024-08-27")]
theorem sup_lt_ord {ι} {f : ι → Ordinal} {c : Ordinal} (hι : #ι < c.cof) :
    (∀ i, f i < c) → sup.{u, u} f < c :=
                      /-
                        ι : Type u
                        f : ι → Ordinal.{u}
                        c : Ordinal.{u}
                        hι : LT.lt (Cardinal.mk ι) c.cof
                        ⊢ LT.lt (Cardinal.lift.{u, u} (Cardinal.mk ι)) c.cof
                      -/
  sup_lt_ord_lift (by rwa [(#ι).lift_id])
                      /-
                        🎉 no goals
                      -/


theorem iSup_lt_lift {ι} {f : ι → Cardinal} {c : Cardinal}
    (hι : Cardinal.lift.{v, u} #ι < c.ord.cof)
    (hf : ∀ i, f i < c) : iSup f < c := by
  /-
    ι : Type u
    f : ι → Cardinal.{max u v}
    c : Cardinal.{max u v}
    hι : LT.lt (Cardinal.lift.{v, u} (Cardinal.mk ι)) c.ord.cof
    hf : ∀ (i : ι), LT.lt (f i) c
    ⊢ LT.lt (iSup f) c
  -/
  rw [← ord_lt_ord, iSup_ord (Cardinal.bddAbove_range _)]
  /-
    ι : Type u
    f : ι → Cardinal.{max u v}
    c : Cardinal.{max u v}
    hι : LT.lt (Cardinal.lift.{v, u} (Cardinal.mk ι)) c.ord.cof
    hf : ∀ (i : ι), LT.lt (f i) c
    ⊢ LT.lt (iSup fun i => (f i).ord) c.ord
  -/
  refine iSup_lt_ord_lift hι fun i => ?_
  /-
    ι : Type u
    f : ι → Cardinal.{max u v}
    c : Cardinal.{max u v}
    hι : LT.lt (Cardinal.lift.{v, u} (Cardinal.mk ι)) c.ord.cof
    hf : ∀ (i : ι), LT.lt (f i) c
    i : ι
    ⊢ LT.lt (f i).ord c.ord
  -/
  rw [ord_lt_ord]
  /-
    ι : Type u
    f : ι → Cardinal.{max u v}
    c : Cardinal.{max u v}
    hι : LT.lt (Cardinal.lift.{v, u} (Cardinal.mk ι)) c.ord.cof
    hf : ∀ (i : ι), LT.lt (f i) c
    i : ι
    ⊢ LT.lt (f i) c
  -/
  apply hf
  /-
    🎉 no goals
  -/


theorem iSup_lt {ι} {f : ι → Cardinal} {c : Cardinal} (hι : #ι < c.ord.cof) :
    (∀ i, f i < c) → iSup f < c :=
                   /-
                     ι : Type u_1
                     f : ι → Cardinal.{u_1}
                     c : Cardinal.{u_1}
                     hι : LT.lt (Cardinal.mk ι) c.ord.cof
                     ⊢ LT.lt (Cardinal.lift.{?u.31773, u_1} (Cardinal.mk ι)) c.ord.cof
                   -/
  iSup_lt_lift (by rwa [(#ι).lift_id])
                   /-
                     🎉 no goals
                   -/


theorem nfpFamily_lt_ord_lift {ι} {f : ι → Ordinal → Ordinal} {c} (hc : ℵ₀ < cof c)
    (hc' : Cardinal.lift.{v, u} #ι < cof c) (hf : ∀ (i), ∀ b < c, f i b < c) {a} (ha : a < c) :
    nfpFamily f a < c := by
  /-
    ι : Type u
    f : ι → Ordinal.{max u v} → Ordinal.{max u v}
    c : Ordinal.{max u v}
    hc : LT.lt Cardinal.aleph0 c.cof
    hc' : LT.lt (Cardinal.lift.{v, u} (Cardinal.mk ι)) c.cof
    hf : ∀ (i : ι) (b : Ordinal.{max u v}), LT.lt b c → LT.lt (f i b) c
    a : Ordinal.{max u v}
    ha : LT.lt a c
    ⊢ LT.lt (Ordinal.nfpFamily f a) c
  -/
  refine iSup_lt_ord_lift ((Cardinal.lift_le.2 (mk_list_le_max ι)).trans_lt ?_) fun l => ?_
    /-
      case refine_1
      ι : Type u
      f : ι → Ordinal.{max u v} → Ordinal.{max u v}
      c : Ordinal.{max u v}
      hc : LT.lt Cardinal.aleph0 c.cof
      hc' : LT.lt (Cardinal.lift.{v, u} (Cardinal.mk ι)) c.cof
      hf : ∀ (i : ι) (b : Ordinal.{max u v}), LT.lt b c → LT.lt (f i b) c
      a : Ordinal.{max u v}
      ha : LT.lt a c
      ⊢ LT.lt (Cardinal.lift.{v, u} (Max.max Cardinal.aleph0 (Cardinal.mk ι))) c.cof
    -/
  · rw [lift_max]
    /-
      case refine_1
      ι : Type u
      f : ι → Ordinal.{max u v} → Ordinal.{max u v}
      c : Ordinal.{max u v}
      hc : LT.lt Cardinal.aleph0 c.cof
      hc' : LT.lt (Cardinal.lift.{v, u} (Cardinal.mk ι)) c.cof
      hf : ∀ (i : ι) (b : Ordinal.{max u v}), LT.lt b c → LT.lt (f i b) c
      a : Ordinal.{max u v}
      ha : LT.lt a c
      ⊢ LT.lt (Max.max (Cardinal.lift.{v, u} Cardinal.aleph0) (Cardinal.lift.{v, u}  …
    -/
    apply max_lt _ hc'
    /-
      ι : Type u
      f : ι → Ordinal.{max u v} → Ordinal.{max u v}
      c : Ordinal.{max u v}
      hc : LT.lt Cardinal.aleph0 c.cof
      hc' : LT.lt (Cardinal.lift.{v, u} (Cardinal.mk ι)) c.cof
      hf : ∀ (i : ι) (b : Ordinal.{max u v}), LT.lt b c → LT.lt (f i b) c
      a : Ordinal.{max u v}
      ha : LT.lt a c
      ⊢ LT.lt (Cardinal.lift.{v, u} Cardinal.aleph0) c.cof
    -/
    rwa [Cardinal.lift_aleph0]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u
      f : ι → Ordinal.{max u v} → Ordinal.{max u v}
      c : Ordinal.{max u v}
      hc : LT.lt Cardinal.aleph0 c.cof
      hc' : LT.lt (Cardinal.lift.{v, u} (Cardinal.mk ι)) c.cof
      hf : ∀ (i : ι) (b : Ordinal.{max u v}), LT.lt b c → LT.lt (f i b) c
      a : Ordinal.{max u v}
      ha : LT.lt a c
      l : List ι
      ⊢ LT.lt (List.foldr f a l) c
    -/
  · induction' l with i l H
      /-
        case refine_2.nil
        ι : Type u
        f : ι → Ordinal.{max u v} → Ordinal.{max u v}
        c : Ordinal.{max u v}
        hc : LT.lt Cardinal.aleph0 c.cof
        hc' : LT.lt (Cardinal.lift.{v, u} (Cardinal.mk ι)) c.cof
        hf : ∀ (i : ι) (b : Ordinal.{max u v}), LT.lt b c → LT.lt (f i b) c
        a : Ordinal.{max u v}
        ha : LT.lt a c
        ⊢ LT.lt (List.foldr f a List.nil) c
      -/
    · exact ha
      /-
        🎉 no goals
      -/
      /-
        case refine_2.cons
        ι : Type u
        f : ι → Ordinal.{max u v} → Ordinal.{max u v}
        c : Ordinal.{max u v}
        hc : LT.lt Cardinal.aleph0 c.cof
        hc' : LT.lt (Cardinal.lift.{v, u} (Cardinal.mk ι)) c.cof
        hf : ∀ (i : ι) (b : Ordinal.{max u v}), LT.lt b c → LT.lt (f i b) c
        a : Ordinal.{max u v}
        ha : LT.lt a c
        i : ι
        l : List ι
        H : LT.lt (List.foldr f a l) c
        ⊢ LT.lt (List.foldr f a (List.cons i l)) c
      -/
    · exact hf _ _ H
      /-
        🎉 no goals
      -/


theorem nfpFamily_lt_ord {ι} {f : ι → Ordinal → Ordinal} {c} (hc : ℵ₀ < cof c) (hc' : #ι < cof c)
    (hf : ∀ (i), ∀ b < c, f i b < c) {a} : a < c → nfpFamily.{u, u} f a < c :=
                               /-
                                 ι : Type u
                                 f : ι → Ordinal.{u} → Ordinal.{u}
                                 c : Ordinal.{u}
                                 hc : LT.lt Cardinal.aleph0 c.cof
                                 hc' : LT.lt (Cardinal.mk ι) c.cof
                                 hf : ∀ (i : ι) (b : Ordinal.{u}), LT.lt b c → LT.lt (f i b) c
                                 a : Ordinal.{u}
                                 ⊢ LT.lt (Cardinal.lift.{?u.33843, u} (Cardinal.mk ι)) c.cof
                               -/
  nfpFamily_lt_ord_lift hc (by rwa [(#ι).lift_id]) hf
                               /-
                                 🎉 no goals
                               -/


set_option linter.deprecated false in
@[deprecated nfpFamily_lt_ord_lift (since := "2024-10-14")]
theorem nfpBFamily_lt_ord_lift {o : Ordinal} {f : ∀ a < o, Ordinal → Ordinal} {c} (hc : ℵ₀ < cof c)
    (hc' : Cardinal.lift.{v, u} o.card < cof c) (hf : ∀ (i hi), ∀ b < c, f i hi b < c) {a} :
    a < c → nfpBFamily.{u, v} o f a < c :=
                               /-
                                 o : Ordinal.{u}
                                 f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v} → Ordinal.{max u v}
                                 c : Ordinal.{max u v}
                                 hc : LT.lt Cardinal.aleph0 c.cof
                                 hc' : LT.lt (Cardinal.lift.{v, u} o.card) c.cof
                                 hf : ∀ (i : Ordinal.{u}) (hi : LT.lt i o) (b : Ordinal.{max u v}), LT.lt b c → …
                                 a : Ordinal.{max u v}
                                 ⊢ LT.lt (Cardinal.lift.{?u.34598, u} (Cardinal.mk o.toType)) c.cof
                               -/
  nfpFamily_lt_ord_lift hc (by rwa [mk_toType]) fun _ => hf _ _
                               /-
                                 🎉 no goals
                               -/


set_option linter.deprecated false in
@[deprecated nfpFamily_lt_ord (since := "2024-10-14")]
theorem nfpBFamily_lt_ord {o : Ordinal} {f : ∀ a < o, Ordinal → Ordinal} {c} (hc : ℵ₀ < cof c)
    (hc' : o.card < cof c) (hf : ∀ (i hi), ∀ b < c, f i hi b < c) {a} :
    a < c → nfpBFamily.{u, u} o f a < c :=
                                /-
                                  o : Ordinal.{u}
                                  f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{u} → Ordinal.{u}
                                  c : Ordinal.{u}
                                  hc : LT.lt Cardinal.aleph0 c.cof
                                  hc' : LT.lt o.card c.cof
                                  hf : ∀ (i : Ordinal.{u}) (hi : LT.lt i o) (b : Ordinal.{u}), LT.lt b c → LT.lt …
                                  a : Ordinal.{u}
                                  ⊢ LT.lt (Cardinal.lift.{u, u} o.card) c.cof
                                -/
  nfpBFamily_lt_ord_lift hc (by rwa [o.card.lift_id]) hf
                                /-
                                  🎉 no goals
                                -/


theorem nfp_lt_ord {f : Ordinal → Ordinal} {c} (hc : ℵ₀ < cof c) (hf : ∀ i < c, f i < c) {a} :
    a < c → nfp f a < c :=
                               /-
                                 f : Ordinal.{u_1} → Ordinal.{u_1}
                                 c : Ordinal.{u_1}
                                 hc : LT.lt Cardinal.aleph0 c.cof
                                 hf : ∀ (i : Ordinal.{u_1}), LT.lt i c → LT.lt (f i) c
                                 a : Ordinal.{u_1}
                                 ⊢ LT.lt (Cardinal.lift.{u_1, 0} (Cardinal.mk Unit)) c.cof
                               -/
  nfpFamily_lt_ord_lift hc (by simpa using Cardinal.one_lt_aleph0.trans hc) fun _ => hf
                               /-
                                 🎉 no goals
                               -/


theorem exists_blsub_cof (o : Ordinal) :
    ∃ f : ∀ a < (cof o).ord, Ordinal, blsub.{u, u} _ f = o := by
  /-
    o : Ordinal.{u}
    ⊢ Exists fun f => Eq (o.cof.ord.blsub f) o
  -/
  rcases exists_lsub_cof o with ⟨ι, f, hf, hι⟩
  /-
    case intro.intro.intro
    o : Ordinal.{u}
    ι : Type u
    f : ι → Ordinal.{u}
    hf : Eq (Ordinal.lsub f) o
    hι : Eq (Cardinal.mk ι) o.cof
    ⊢ Exists fun f => Eq (o.cof.ord.blsub f) o
  -/
  rcases Cardinal.ord_eq ι with ⟨r, hr, hι'⟩
  /-
    case intro.intro.intro.intro.intro
    o : Ordinal.{u}
    ι : Type u
    f : ι → Ordinal.{u}
    hf : Eq (Ordinal.lsub f) o
    hι : Eq (Cardinal.mk ι) o.cof
    r : ι → ι → Prop
    hr : IsWellOrder ι r
    hι' : Eq (Cardinal.mk ι).ord (Ordinal.type r)
    ⊢ Exists fun f => Eq (o.cof.ord.blsub f) o
  -/
  rw [← @blsub_eq_lsub' ι r hr] at hf
  /-
    case intro.intro.intro.intro.intro
    o : Ordinal.{u}
    ι : Type u
    f : ι → Ordinal.{u}
    hι : Eq (Cardinal.mk ι) o.cof
    r : ι → ι → Prop
    hr : IsWellOrder ι r
    hf : Eq ((Ordinal.type r).blsub (Ordinal.bfamilyOfFamily' r f)) o
    hι' : Eq (Cardinal.mk ι).ord (Ordinal.type r)
    ⊢ Exists fun f => Eq (o.cof.ord.blsub f) o
  -/
  rw [← hι, hι']
  /-
    case intro.intro.intro.intro.intro
    o : Ordinal.{u}
    ι : Type u
    f : ι → Ordinal.{u}
    hι : Eq (Cardinal.mk ι) o.cof
    r : ι → ι → Prop
    hr : IsWellOrder ι r
    hf : Eq ((Ordinal.type r).blsub (Ordinal.bfamilyOfFamily' r f)) o
    hι' : Eq (Cardinal.mk ι).ord (Ordinal.type r)
    ⊢ Exists fun f => Eq ((Ordinal.type r).blsub f) o
  -/
  exact ⟨_, hf⟩
  /-
    🎉 no goals
  -/


theorem le_cof_iff_blsub {b : Ordinal} {a : Cardinal} :
    a ≤ cof b ↔ ∀ {o} (f : ∀ a < o, Ordinal), blsub.{u, u} o f = b → a ≤ o.card :=
  le_cof_iff_lsub.trans
                        /-
                          b : Ordinal.{u}
                          a : Cardinal.{u}
                          H : ∀ {ι : Type u} (f : ι → Ordinal.{u}), Eq (Ordinal.lsub f) b → LE.le a (Car …
                          o : Ordinal.{u}
                          f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{u}
                          hf : Eq (o.blsub f) b
                          ⊢ LE.le a o.card
                        -/
    ⟨fun H o f hf => by simpa using H _ hf, fun H ι f hf => by
                        /-
                          🎉 no goals
                        -/
      /-
        b : Ordinal.{u}
        a : Cardinal.{u}
        H : ∀ {o : Ordinal.{u}} (f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{u}), Eq  …
        ι : Type u
        f : ι → Ordinal.{u}
        hf : Eq (Ordinal.lsub f) b
        ⊢ LE.le a (Cardinal.mk ι)
      -/
      rcases Cardinal.ord_eq ι with ⟨r, hr, hι'⟩
      /-
        case intro.intro
        b : Ordinal.{u}
        a : Cardinal.{u}
        H : ∀ {o : Ordinal.{u}} (f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{u}), Eq  …
        ι : Type u
        f : ι → Ordinal.{u}
        hf : Eq (Ordinal.lsub f) b
        r : ι → ι → Prop
        hr : IsWellOrder ι r
        hι' : Eq (Cardinal.mk ι).ord (Ordinal.type r)
        ⊢ LE.le a (Cardinal.mk ι)
      -/
      rw [← @blsub_eq_lsub' ι r hr] at hf
      /-
        case intro.intro
        b : Ordinal.{u}
        a : Cardinal.{u}
        H : ∀ {o : Ordinal.{u}} (f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{u}), Eq  …
        ι : Type u
        f : ι → Ordinal.{u}
        r : ι → ι → Prop
        hr : IsWellOrder ι r
        hf : Eq ((Ordinal.type r).blsub (Ordinal.bfamilyOfFamily' r f)) b
        hι' : Eq (Cardinal.mk ι).ord (Ordinal.type r)
        ⊢ LE.le a (Cardinal.mk ι)
      -/
      simpa using H _ hf⟩
      /-
        🎉 no goals
      -/


theorem cof_blsub_le_lift {o} (f : ∀ a < o, Ordinal) :
    cof (blsub.{u, v} o f) ≤ Cardinal.lift.{v, u} o.card := by
  /-
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
    ⊢ LE.le (o.blsub f).cof (Cardinal.lift.{v, u} o.card)
  -/
  rw [← mk_toType o]
  /-
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
    ⊢ LE.le (o.blsub f).cof (Cardinal.lift.{v, u} (Cardinal.mk o.toType))
  -/
  exact cof_lsub_le_lift _
  /-
    🎉 no goals
  -/


theorem cof_blsub_le {o} (f : ∀ a < o, Ordinal) : cof (blsub.{u, u} o f) ≤ o.card := by
  /-
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{u}
    ⊢ LE.le (o.blsub f).cof o.card
  -/
  rw [← o.card.lift_id]
  /-
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{u}
    ⊢ LE.le (o.blsub f).cof (Cardinal.lift.{u, u} o.card)
  -/
  exact cof_blsub_le_lift f
  /-
    🎉 no goals
  -/


theorem blsub_lt_ord_lift {o : Ordinal.{u}} {f : ∀ a < o, Ordinal} {c : Ordinal}
    (ho : Cardinal.lift.{v, u} o.card < c.cof) (hf : ∀ i hi, f i hi < c) : blsub.{u, v} o f < c :=
  lt_of_le_of_ne (blsub_le hf) fun h =>
                  /-
                    o : Ordinal.{u}
                    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
                    c : Ordinal.{max u v}
                    ho : LT.lt (Cardinal.lift.{v, u} o.card) c.cof
                    hf : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), LT.lt (f i hi) c
                    h : Eq (o.blsub f) c
                    ⊢ LE.le c.cof (Cardinal.lift.{v, u} o.card)
                  -/
    ho.not_le (by simpa [← iSup_ord, hf, h] using cof_blsub_le_lift.{u, v} f)
                  /-
                    🎉 no goals
                  -/


theorem blsub_lt_ord {o : Ordinal} {f : ∀ a < o, Ordinal} {c : Ordinal} (ho : o.card < c.cof)
    (hf : ∀ i hi, f i hi < c) : blsub.{u, u} o f < c :=
                        /-
                          o : Ordinal.{u}
                          f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{u}
                          c : Ordinal.{u}
                          ho : LT.lt o.card c.cof
                          hf : ∀ (i : Ordinal.{u}) (hi : LT.lt i o), LT.lt (f i hi) c
                          ⊢ LT.lt (Cardinal.lift.{u, u} o.card) c.cof
                        -/
  blsub_lt_ord_lift (by rwa [o.card.lift_id]) hf
                        /-
                          🎉 no goals
                        -/


theorem cof_bsup_le_lift {o : Ordinal} {f : ∀ a < o, Ordinal} (H : ∀ i h, f i h < bsup.{u, v} o f) :
    cof (bsup.{u, v} o f) ≤ Cardinal.lift.{v, u} o.card := by
  /-
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
    H : ∀ (i : Ordinal.{u}) (h : LT.lt i o), LT.lt (f i h) (o.bsup f)
    ⊢ LE.le (o.bsup f).cof (Cardinal.lift.{v, u} o.card)
  -/
  rw [← bsup_eq_blsub_iff_lt_bsup.{u, v}] at H
  /-
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
    H : Eq (o.bsup f) (o.blsub f)
    ⊢ LE.le (o.bsup f).cof (Cardinal.lift.{v, u} o.card)
  -/
  rw [H]
  /-
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
    H : Eq (o.bsup f) (o.blsub f)
    ⊢ LE.le (o.blsub f).cof (Cardinal.lift.{v, u} o.card)
  -/
  exact cof_blsub_le_lift.{u, v} f
  /-
    🎉 no goals
  -/


theorem cof_bsup_le {o : Ordinal} {f : ∀ a < o, Ordinal} :
    (∀ i h, f i h < bsup.{u, u} o f) → cof (bsup.{u, u} o f) ≤ o.card := by
  /-
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{u}
    ⊢ (∀ (i : Ordinal.{u}) (h : LT.lt i o), LT.lt (f i h) (o.bsup f)) → LE.le (o.b …
  -/
  rw [← o.card.lift_id]
  /-
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{u}
    ⊢ (∀ (i : Ordinal.{u}) (h : LT.lt i o), LT.lt (f i h) (o.bsup f)) → LE.le (o.b …
  -/
  exact cof_bsup_le_lift
  /-
    🎉 no goals
  -/


theorem bsup_lt_ord_lift {o : Ordinal} {f : ∀ a < o, Ordinal} {c : Ordinal}
    (ho : Cardinal.lift.{v, u} o.card < c.cof) (hf : ∀ i hi, f i hi < c) : bsup.{u, v} o f < c :=
  (bsup_le_blsub f).trans_lt (blsub_lt_ord_lift ho hf)


theorem bsup_lt_ord {o : Ordinal} {f : ∀ a < o, Ordinal} {c : Ordinal} (ho : o.card < c.cof) :
    (∀ i hi, f i hi < c) → bsup.{u, u} o f < c :=
                       /-
                         o : Ordinal.{u}
                         f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{u}
                         c : Ordinal.{u}
                         ho : LT.lt o.card c.cof
                         ⊢ LT.lt (Cardinal.lift.{u, u} o.card) c.cof
                       -/
  bsup_lt_ord_lift (by rwa [o.card.lift_id])
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem cof_zero : cof 0 = 0 := by
  /-
    ⊢ Eq (Ordinal.cof 0) 0
  -/
  refine LE.le.antisymm  ?_ (Cardinal.zero_le _)
  /-
    ⊢ LE.le (Ordinal.cof 0) 0
  -/
  rw [← card_zero]
  /-
    ⊢ LE.le (Ordinal.cof 0) (Ordinal.card 0)
  -/
  exact cof_le_card 0
  /-
    🎉 no goals
  -/


@[simp]
theorem cof_eq_zero {o} : cof o = 0 ↔ o = 0 :=
  ⟨inductionOn o fun _ r _ z =>
      let ⟨_, hl, e⟩ := cof_eq r
      type_eq_zero_iff_isEmpty.2 <|
        ⟨fun a =>
          let ⟨_, h, _⟩ := hl a
          (mk_eq_zero_iff.1 (e.trans z)).elim' ⟨_, h⟩⟩,
                /-
                  o : Ordinal.{u_1}
                  e : Eq o 0
                  ⊢ Eq o.cof 0
                -/
    fun e => by simp [e]⟩
                /-
                  🎉 no goals
                -/


theorem cof_ne_zero {o} : cof o ≠ 0 ↔ o ≠ 0 :=
  cof_eq_zero.not


@[simp]
theorem cof_succ (o) : cof (succ o) = 1 := by
  /-
    o : Ordinal.{u_1}
    ⊢ Eq (Order.succ o).cof 1
  -/
  apply le_antisymm
    /-
      case a
      o : Ordinal.{u_1}
      ⊢ LE.le (Order.succ o).cof 1
    -/
  · refine inductionOn o fun α r _ => ?_
    /-
      case a
      o : Ordinal.{u_1}
      α : Type u_1
      r : α → α → Prop
      x✝ : IsWellOrder α r
      ⊢ LE.le (Order.succ (Ordinal.type r)).cof 1
    -/
    change cof (type _) ≤ _
    /-
      case a
      o : Ordinal.{u_1}
      α : Type u_1
      r : α → α → Prop
      x✝ : IsWellOrder α r
      ⊢ LE.le (Ordinal.type (Sum.Lex r EmptyRelation)).cof 1
    -/
    rw [← (_ : #_ = 1)]
      /-
        case a
        o : Ordinal.{u_1}
        α : Type u_1
        r : α → α → Prop
        x✝ : IsWellOrder α r
        ⊢ LE.le (Ordinal.type (Sum.Lex r EmptyRelation)).cof (Cardinal.mk ?m.46466)
      -/
    · apply cof_type_le
      /-
        case a.h
        o : Ordinal.{u_1}
        α : Type u_1
        r : α → α → Prop
        x✝ : IsWellOrder α r
        ⊢ Set.Unbounded (Sum.Lex r EmptyRelation) ?a.S✝
      -/
      refine fun a => ⟨Sum.inr PUnit.unit, Set.mem_singleton _, ?_⟩
      /-
        case a.h
        o : Ordinal.{u_1}
        α : Type u_1
        r : α → α → Prop
        x✝ : IsWellOrder α r
        a : Sum α PUnit.{u_1 + 1}
        ⊢ Not (Sum.Lex r EmptyRelation (Sum.inr PUnit.unit) a)
      -/
                                     /-
                                       🎉 no goals
                                     -/
      rcases a with (a | ⟨⟨⟨⟩⟩⟩) <;> simp [EmptyRelation]
                                     /-
                                       🎉 no goals
                                     -/
      /-
        o : Ordinal.{u_1}
        α : Type u_1
        r : α → α → Prop
        x✝ : IsWellOrder α r
        ⊢ Eq (Cardinal.mk ↑(Singleton.singleton (Sum.inr PUnit.unit))) 1
      -/
    · rw [Cardinal.mk_fintype, Set.card_singleton]
      /-
        o : Ordinal.{u_1}
        α : Type u_1
        r : α → α → Prop
        x✝ : IsWellOrder α r
        ⊢ Eq (↑1) 1
      -/
      simp
      /-
        🎉 no goals
      -/
    /-
      case a
      o : Ordinal.{u_1}
      ⊢ LE.le 1 (Order.succ o).cof
    -/
  · rw [← Cardinal.succ_zero, succ_le_iff]
    simpa [lt_iff_le_and_ne, Cardinal.zero_le] using fun h =>
      succ_ne_zero o (cof_eq_zero.1 (Eq.symm h))


@[simp]
theorem cof_eq_one_iff_is_succ {o} : cof.{u} o = 1 ↔ ∃ a, o = succ a :=
  ⟨inductionOn o fun α r _ z => by
      /-
        o : Ordinal.{u}
        α : Type u
        r : α → α → Prop
        x✝ : IsWellOrder α r
        z : Eq (Ordinal.type r).cof 1
        ⊢ Exists fun a => Eq (Ordinal.type r) (Order.succ a)
      -/
      rcases cof_eq r with ⟨S, hl, e⟩; rw [z] at e
      /-
        case intro.intro
        o : Ordinal.{u}
        α : Type u
        r : α → α → Prop
        x✝ : IsWellOrder α r
        z : Eq (Ordinal.type r).cof 1
        S : Set α
        hl : Set.Unbounded r S
        e : Eq (Cardinal.mk ↑S) 1
        ⊢ Exists fun a => Eq (Ordinal.type r) (Order.succ a)
      -/
      cases' mk_ne_zero_iff.1 (by rw [e]; exact one_ne_zero) with a
      refine
        ⟨typein r a,
          Eq.symm <|
            Quotient.sound
              ⟨RelIso.ofSurjective (RelEmbedding.ofMonotone ?_ fun x y => ?_) fun x => ?_⟩⟩
        /-
          case intro.intro.intro.refine_1
          o : Ordinal.{u}
          α : Type u
          r : α → α → Prop
          x✝ : IsWellOrder α r
          z : Eq (Ordinal.type r).cof 1
          S : Set α
          hl : Set.Unbounded r S
          e : Eq (Cardinal.mk ↑S) 1
          a : ↑S
          ⊢ Sum (Subtype fun b => r b ↑a) PUnit.{u + 1} → α
        -/
      · apply Sum.rec <;> [exact Subtype.val; exact fun _ => a]
        /-
          🎉 no goals
        -/
        /-
          case intro.intro.intro.refine_2
          o : Ordinal.{u}
          α : Type u
          r : α → α → Prop
          x✝ : IsWellOrder α r
          z : Eq (Ordinal.type r).cof 1
          S : Set α
          hl : Set.Unbounded r S
          e : Eq (Cardinal.mk ↑S) 1
          a : ↑S
          x y : Sum (Subtype fun b => r b ↑a) PUnit.{u + 1}
          ⊢ Sum.Lex (Subrel r (setOf fun b => r b ↑a)) EmptyRelation x y → r (Sum.rec Su …
        -/
      · rcases x with (x | ⟨⟨⟨⟩⟩⟩) <;> rcases y with (y | ⟨⟨⟨⟩⟩⟩) <;>
          /-
            case intro.intro.intro.refine_2.inl.inl
            o : Ordinal.{u}
            α : Type u
            r : α → α → Prop
            x✝ : IsWellOrder α r
            z : Eq (Ordinal.type r).cof 1
            S : Set α
            hl : Set.Unbounded r S
            e : Eq (Cardinal.mk ↑S) 1
            a : ↑S
            x y : Subtype fun b => r b ↑a
            ⊢ Sum.Lex (Subrel r (setOf fun b => r b ↑a)) EmptyRelation (Sum.inl x) (Sum.in …
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
          simp [Subrel, Order.Preimage, EmptyRelation]
          /-
            🎉 no goals
          -/
        /-
          case intro.intro.intro.refine_2.inl.inr.unit
          o : Ordinal.{u}
          α : Type u
          r : α → α → Prop
          x✝ : IsWellOrder α r
          z : Eq (Ordinal.type r).cof 1
          S : Set α
          hl : Set.Unbounded r S
          e : Eq (Cardinal.mk ↑S) 1
          a : ↑S
          x : Subtype fun b => r b ↑a
          ⊢ r ↑x ↑a
        -/
        exact x.2
        /-
          🎉 no goals
        -/
      · suffices r x a ∨ ∃ _ : PUnit.{u}, ↑a = x by
          convert this
          dsimp [RelEmbedding.ofMonotone]; simp
        /-
          case intro.intro.intro.refine_3
          o : Ordinal.{u}
          α : Type u
          r : α → α → Prop
          x✝ : IsWellOrder α r
          z : Eq (Ordinal.type r).cof 1
          S : Set α
          hl : Set.Unbounded r S
          e : Eq (Cardinal.mk ↑S) 1
          a : ↑S
          x : α
          ⊢ Or (r x ↑a) (Exists fun x_1 => Eq (↑a) x)
        -/
        rcases trichotomous_of r x a with (h | h | h)
          /-
            case intro.intro.intro.refine_3.inl
            o : Ordinal.{u}
            α : Type u
            r : α → α → Prop
            x✝ : IsWellOrder α r
            z : Eq (Ordinal.type r).cof 1
            S : Set α
            hl : Set.Unbounded r S
            e : Eq (Cardinal.mk ↑S) 1
            a : ↑S
            x : α
            h : r x ↑a
            ⊢ Or (r x ↑a) (Exists fun x_1 => Eq (↑a) x)
          -/
        · exact Or.inl h
          /-
            🎉 no goals
          -/
          /-
            case intro.intro.intro.refine_3.inr.inl
            o : Ordinal.{u}
            α : Type u
            r : α → α → Prop
            x✝ : IsWellOrder α r
            z : Eq (Ordinal.type r).cof 1
            S : Set α
            hl : Set.Unbounded r S
            e : Eq (Cardinal.mk ↑S) 1
            a : ↑S
            x : α
            h : Eq x ↑a
            ⊢ Or (r x ↑a) (Exists fun x_1 => Eq (↑a) x)
          -/
        · exact Or.inr ⟨PUnit.unit, h.symm⟩
          /-
            🎉 no goals
          -/
          /-
            case intro.intro.intro.refine_3.inr.inr
            o : Ordinal.{u}
            α : Type u
            r : α → α → Prop
            x✝ : IsWellOrder α r
            z : Eq (Ordinal.type r).cof 1
            S : Set α
            hl : Set.Unbounded r S
            e : Eq (Cardinal.mk ↑S) 1
            a : ↑S
            x : α
            h : r (↑a) x
            ⊢ Or (r x ↑a) (Exists fun x_1 => Eq (↑a) x)
          -/
        · rcases hl x with ⟨a', aS, hn⟩
          /-
            case intro.intro.intro.refine_3.inr.inr.intro.intro
            o : Ordinal.{u}
            α : Type u
            r : α → α → Prop
            x✝ : IsWellOrder α r
            z : Eq (Ordinal.type r).cof 1
            S : Set α
            hl : Set.Unbounded r S
            e : Eq (Cardinal.mk ↑S) 1
            a : ↑S
            x : α
            h : r (↑a) x
            a' : α
            aS : Membership.mem S a'
            hn : Not (r a' x)
            ⊢ Or (r x ↑a) (Exists fun x_1 => Eq (↑a) x)
          -/
          refine absurd h ?_
          /-
            case intro.intro.intro.refine_3.inr.inr.intro.intro
            o : Ordinal.{u}
            α : Type u
            r : α → α → Prop
            x✝ : IsWellOrder α r
            z : Eq (Ordinal.type r).cof 1
            S : Set α
            hl : Set.Unbounded r S
            e : Eq (Cardinal.mk ↑S) 1
            a : ↑S
            x : α
            h : r (↑a) x
            a' : α
            aS : Membership.mem S a'
            hn : Not (r a' x)
            ⊢ Not (r (↑a) x)
          -/
          convert hn
          /-
            case h.e'_1.h.e'_1
            o : Ordinal.{u}
            α : Type u
            r : α → α → Prop
            x✝ : IsWellOrder α r
            z : Eq (Ordinal.type r).cof 1
            S : Set α
            hl : Set.Unbounded r S
            e : Eq (Cardinal.mk ↑S) 1
            a : ↑S
            x : α
            h : r (↑a) x
            a' : α
            aS : Membership.mem S a'
            hn : Not (r a' x)
            ⊢ Eq (↑a) a'
          -/
          change (a : α) = ↑(⟨a', aS⟩ : S)
          /-
            case h.e'_1.h.e'_1
            o : Ordinal.{u}
            α : Type u
            r : α → α → Prop
            x✝ : IsWellOrder α r
            z : Eq (Ordinal.type r).cof 1
            S : Set α
            hl : Set.Unbounded r S
            e : Eq (Cardinal.mk ↑S) 1
            a : ↑S
            x : α
            h : r (↑a) x
            a' : α
            aS : Membership.mem S a'
            hn : Not (r a' x)
            ⊢ Eq ↑a ↑⟨a', aS⟩
          -/
          have := le_one_iff_subsingleton.1 (le_of_eq e)
          /-
            case h.e'_1.h.e'_1
            o : Ordinal.{u}
            α : Type u
            r : α → α → Prop
            x✝ : IsWellOrder α r
            z : Eq (Ordinal.type r).cof 1
            S : Set α
            hl : Set.Unbounded r S
            e : Eq (Cardinal.mk ↑S) 1
            a : ↑S
            x : α
            h : r (↑a) x
            a' : α
            aS : Membership.mem S a'
            hn : Not (r a' x)
            this : Subsingleton ↑S
            ⊢ Eq ↑a ↑⟨a', aS⟩
          -/
          congr!,
          /-
            🎉 no goals
          -/
                     /-
                       o : Ordinal.{u}
                       x✝ : Exists fun a => Eq o (Order.succ a)
                       a : Ordinal.{u}
                       e : Eq o (Order.succ a)
                       ⊢ Eq o.cof 1
                     -/
    fun ⟨a, e⟩ => by simp [e]⟩
                     /-
                       🎉 no goals
                     -/


/-- A fundamental sequence for `a` is an increasing sequence of length `o = cof a` that converges at
    `a`. We provide `o` explicitly in order to avoid type rewrites. -/
def IsFundamentalSequence (a o : Ordinal.{u}) (f : ∀ b < o, Ordinal.{u}) : Prop :=
  o ≤ a.cof.ord ∧ (∀ {i j} (hi hj), i < j → f i hi < f j hj) ∧ blsub.{u, u} o f = a


protected theorem cof_eq (hf : IsFundamentalSequence a o f) : a.cof.ord = o :=
  hf.1.antisymm' <| by
    /-
      a o : Ordinal.{u}
      f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
      hf : a.IsFundamentalSequence o f
      ⊢ LE.le a.cof.ord o
    -/
    rw [← hf.2.2]
    /-
      a o : Ordinal.{u}
      f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
      hf : a.IsFundamentalSequence o f
      ⊢ LE.le (o.blsub f).cof.ord o
    -/
    exact (ord_le_ord.2 (cof_blsub_le f)).trans (ord_card_le o)
    /-
      🎉 no goals
    -/


protected theorem strict_mono (hf : IsFundamentalSequence a o f) {i j} :
    ∀ hi hj, i < j → f i hi < f j hj :=
  hf.2.1


theorem blsub_eq (hf : IsFundamentalSequence a o f) : blsub.{u, u} o f = a :=
  hf.2.2


theorem ord_cof (hf : IsFundamentalSequence a o f) :
                                                                       /-
                                                                         α : Type u
                                                                         β : Type v
                                                                         r : α → α → Prop
                                                                         s : β → β → Prop
                                                                         a o : Ordinal.{u}
                                                                         f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
                                                                         hf : a.IsFundamentalSequence o f
                                                                         i : Ordinal.{u}
                                                                         hi : LT.lt i a.cof.ord
                                                                         ⊢ LE.le a.cof.ord o
                                                                       -/
    IsFundamentalSequence a a.cof.ord fun i hi => f i (hi.trans_le (by rw [hf.cof_eq])) := by
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  /-
    a o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
    hf : a.IsFundamentalSequence o f
    ⊢ a.IsFundamentalSequence a.cof.ord fun i hi => f i ⋯
  -/
  have H := hf.cof_eq
  /-
    a o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
    hf : a.IsFundamentalSequence o f
    H : Eq a.cof.ord o
    ⊢ a.IsFundamentalSequence a.cof.ord fun i hi => f i ⋯
  -/
  subst H
  /-
    a : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b a.cof.ord → Ordinal.{u}
    hf : a.IsFundamentalSequence a.cof.ord f
    ⊢ a.IsFundamentalSequence a.cof.ord fun i hi => f i ⋯
  -/
  exact hf
  /-
    🎉 no goals
  -/


theorem id_of_le_cof (h : o ≤ o.cof.ord) : IsFundamentalSequence o o fun a _ => a :=
  ⟨h, @fun _ _ _ _ => id, blsub_id o⟩


protected theorem zero {f : ∀ b < (0 : Ordinal), Ordinal} : IsFundamentalSequence 0 0 f :=
      /-
        f : (b : Ordinal.{u_1}) → LT.lt b 0 → Ordinal.{u_1}
        ⊢ LE.le 0 (Ordinal.cof 0).ord
      -/
  ⟨by rw [cof_zero, ord_zero], @fun i _ hi => (Ordinal.not_lt_zero i hi).elim, blsub_zero f⟩
      /-
        🎉 no goals
      -/


protected theorem succ : IsFundamentalSequence (succ o) 1 fun _ _ => o := by
  /-
    o : Ordinal.{u}
    ⊢ (Order.succ o).IsFundamentalSequence 1 fun x x => o
  -/
  refine ⟨?_, @fun i j hi hj h => ?_, blsub_const Ordinal.one_ne_zero o⟩
    /-
      case refine_1
      o : Ordinal.{u}
      ⊢ LE.le 1 (Order.succ o).cof.ord
    -/
  · rw [cof_succ, ord_one]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      o i j : Ordinal.{u}
      hi : LT.lt i 1
      hj : LT.lt j 1
      h : LT.lt i j
      ⊢ LT.lt ((fun x x => o) i hi) ((fun x x => o) j hj)
    -/
  · rw [lt_one_iff_zero] at hi hj
    /-
      case refine_2
      o i j : Ordinal.{u}
      hi✝ : LT.lt i 1
      hi : Eq i 0
      hj✝ : LT.lt j 1
      hj : Eq j 0
      h : LT.lt i j
      ⊢ LT.lt ((fun x x => o) i hi✝) ((fun x x => o) j hj✝)
    -/
    rw [hi, hj] at h
    /-
      case refine_2
      o i j : Ordinal.{u}
      hi✝ : LT.lt i 1
      hi : Eq i 0
      hj✝ : LT.lt j 1
      hj : Eq j 0
      h : LT.lt 0 0
      ⊢ LT.lt ((fun x x => o) i hi✝) ((fun x x => o) j hj✝)
    -/
    exact h.false.elim
    /-
      🎉 no goals
    -/


protected theorem monotone (hf : IsFundamentalSequence a o f) {i j : Ordinal} (hi : i < o)
    (hj : j < o) (hij : i ≤ j) : f i hi ≤ f j hj := by
  /-
    a o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
    hf : a.IsFundamentalSequence o f
    i j : Ordinal.{u}
    hi : LT.lt i o
    hj : LT.lt j o
    hij : LE.le i j
    ⊢ LE.le (f i hi) (f j hj)
  -/
  rcases lt_or_eq_of_le hij with (hij | rfl)
    /-
      case inl
      a o : Ordinal.{u}
      f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
      hf : a.IsFundamentalSequence o f
      i j : Ordinal.{u}
      hi : LT.lt i o
      hj : LT.lt j o
      hij✝ : LE.le i j
      hij : LT.lt i j
      ⊢ LE.le (f i hi) (f j hj)
    -/
  · exact (hf.2.1 hi hj hij).le
    /-
      🎉 no goals
    -/
    /-
      case inr
      a o : Ordinal.{u}
      f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
      hf : a.IsFundamentalSequence o f
      i : Ordinal.{u}
      hi hj : LT.lt i o
      hij : LE.le i i
      ⊢ LE.le (f i hi) (f i hj)
    -/
  · rfl
    /-
      🎉 no goals
    -/


theorem trans {a o o' : Ordinal.{u}} {f : ∀ b < o, Ordinal.{u}} (hf : IsFundamentalSequence a o f)
    {g : ∀ b < o', Ordinal.{u}} (hg : IsFundamentalSequence o o' g) :
    IsFundamentalSequence a o' fun i hi =>
                     /-
                       α : Type u
                       β : Type v
                       r : α → α → Prop
                       s : β → β → Prop
                       a✝ o✝ : Ordinal.{u}
                       f✝ : (b : Ordinal.{u}) → LT.lt b o✝ → Ordinal.{u}
                       a o o' : Ordinal.{u}
                       f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
                       hf : a.IsFundamentalSequence o f
                       g : (b : Ordinal.{u}) → LT.lt b o' → Ordinal.{u}
                       hg : o.IsFundamentalSequence o' g
                       i : Ordinal.{u}
                       hi : LT.lt i o'
                       ⊢ LT.lt (g i hi) o
                     -/
      f (g i hi) (by rw [← hg.2.2]; apply lt_blsub) := by
                                    /-
                                      🎉 no goals
                                    -/
  /-
    a o o' : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
    hf : a.IsFundamentalSequence o f
    g : (b : Ordinal.{u}) → LT.lt b o' → Ordinal.{u}
    hg : o.IsFundamentalSequence o' g
    ⊢ a.IsFundamentalSequence o' fun i hi => f (g i hi) ⋯
  -/
  refine ⟨?_, @fun i j _ _ h => hf.2.1 _ _ (hg.2.1 _ _ h), ?_⟩
    /-
      case refine_1
      a o o' : Ordinal.{u}
      f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
      hf : a.IsFundamentalSequence o f
      g : (b : Ordinal.{u}) → LT.lt b o' → Ordinal.{u}
      hg : o.IsFundamentalSequence o' g
      ⊢ LE.le o' a.cof.ord
    -/
  · rw [hf.cof_eq]
    /-
      case refine_1
      a o o' : Ordinal.{u}
      f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
      hf : a.IsFundamentalSequence o f
      g : (b : Ordinal.{u}) → LT.lt b o' → Ordinal.{u}
      hg : o.IsFundamentalSequence o' g
      ⊢ LE.le o' o
    -/
    exact hg.1.trans (ord_cof_le o)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      a o o' : Ordinal.{u}
      f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
      hf : a.IsFundamentalSequence o f
      g : (b : Ordinal.{u}) → LT.lt b o' → Ordinal.{u}
      hg : o.IsFundamentalSequence o' g
      ⊢ Eq (o'.blsub fun i hi => f (g i hi) ⋯) a
    -/
  · rw [@blsub_comp.{u, u, u} o _ f (@IsFundamentalSequence.monotone _ _ f hf)]
      /-
        case refine_2
        a o o' : Ordinal.{u}
        f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
        hf : a.IsFundamentalSequence o f
        g : (b : Ordinal.{u}) → LT.lt b o' → Ordinal.{u}
        hg : o.IsFundamentalSequence o' g
        ⊢ Eq (o.blsub f) a
      -/
    · exact hf.2.2
      /-
        🎉 no goals
      -/
      /-
        case refine_2.hg
        a o o' : Ordinal.{u}
        f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
        hf : a.IsFundamentalSequence o f
        g : (b : Ordinal.{u}) → LT.lt b o' → Ordinal.{u}
        hg : o.IsFundamentalSequence o' g
        ⊢ Eq (o'.blsub g) o
      -/
    · exact hg.2.2
      /-
        🎉 no goals
      -/


protected theorem lt {a o : Ordinal} {s : Π p < o, Ordinal}
    (h : IsFundamentalSequence a o s) {p : Ordinal} (hp : p < o) : s p hp < a :=
  h.blsub_eq ▸ lt_blsub s p hp


/-- Every ordinal has a fundamental sequence. -/
theorem exists_fundamental_sequence (a : Ordinal.{u}) :
    ∃ f, IsFundamentalSequence a a.cof.ord f := by
  suffices h : ∃ o f, IsFundamentalSequence a o f by
    rcases h with ⟨o, f, hf⟩
    exact ⟨_, hf.ord_cof⟩
  /-
    a : Ordinal.{u}
    ⊢ Exists fun o => Exists fun f => a.IsFundamentalSequence o f
  -/
  rcases exists_lsub_cof a with ⟨ι, f, hf, hι⟩
  /-
    case intro.intro.intro
    a : Ordinal.{u}
    ι : Type u
    f : ι → Ordinal.{u}
    hf : Eq (Ordinal.lsub f) a
    hι : Eq (Cardinal.mk ι) a.cof
    ⊢ Exists fun o => Exists fun f => a.IsFundamentalSequence o f
  -/
  rcases ord_eq ι with ⟨r, wo, hr⟩
  /-
    case intro.intro.intro.intro.intro
    a : Ordinal.{u}
    ι : Type u
    f : ι → Ordinal.{u}
    hf : Eq (Ordinal.lsub f) a
    hι : Eq (Cardinal.mk ι) a.cof
    r : ι → ι → Prop
    wo : IsWellOrder ι r
    hr : Eq (Cardinal.mk ι).ord (Ordinal.type r)
    ⊢ Exists fun o => Exists fun f => a.IsFundamentalSequence o f
  -/
  haveI := wo
  /-
    case intro.intro.intro.intro.intro
    a : Ordinal.{u}
    ι : Type u
    f : ι → Ordinal.{u}
    hf : Eq (Ordinal.lsub f) a
    hι : Eq (Cardinal.mk ι) a.cof
    r : ι → ι → Prop
    wo : IsWellOrder ι r
    hr : Eq (Cardinal.mk ι).ord (Ordinal.type r)
    this : IsWellOrder ι r
    ⊢ Exists fun o => Exists fun f => a.IsFundamentalSequence o f
  -/
  let r' := Subrel r { i | ∀ j, r j i → f j < f i }
  /-
    case intro.intro.intro.intro.intro
    a : Ordinal.{u}
    ι : Type u
    f : ι → Ordinal.{u}
    hf : Eq (Ordinal.lsub f) a
    hι : Eq (Cardinal.mk ι) a.cof
    r : ι → ι → Prop
    wo : IsWellOrder ι r
    hr : Eq (Cardinal.mk ι).ord (Ordinal.type r)
    this : IsWellOrder ι r
    r' : ↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i)) → ↑(setOf fun i => …
    ⊢ Exists fun o => Exists fun f => a.IsFundamentalSequence o f
  -/
  let hrr' : r' ↪r r := Subrel.relEmbedding _ _
  /-
    case intro.intro.intro.intro.intro
    a : Ordinal.{u}
    ι : Type u
    f : ι → Ordinal.{u}
    hf : Eq (Ordinal.lsub f) a
    hι : Eq (Cardinal.mk ι) a.cof
    r : ι → ι → Prop
    wo : IsWellOrder ι r
    hr : Eq (Cardinal.mk ι).ord (Ordinal.type r)
    this : IsWellOrder ι r
    r' : ↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i)) → ↑(setOf fun i => …
    hrr' : RelEmbedding r' r := Subrel.relEmbedding r (setOf fun i => ∀ (j : ι), r …
    ⊢ Exists fun o => Exists fun f => a.IsFundamentalSequence o f
  -/
  haveI := hrr'.isWellOrder
  refine
    ⟨_, _, hrr'.ordinal_type_le.trans ?_, @fun i j _ h _ => (enum r' ⟨j, h⟩).prop _ ?_,
      le_antisymm (blsub_le fun i hi => lsub_le_iff.1 hf.le _) ?_⟩
    /-
      case intro.intro.intro.intro.intro.refine_1
      a : Ordinal.{u}
      ι : Type u
      f : ι → Ordinal.{u}
      hf : Eq (Ordinal.lsub f) a
      hι : Eq (Cardinal.mk ι) a.cof
      r : ι → ι → Prop
      wo : IsWellOrder ι r
      hr : Eq (Cardinal.mk ι).ord (Ordinal.type r)
      this✝ : IsWellOrder ι r
      r' : ↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i)) → ↑(setOf fun i => …
      hrr' : RelEmbedding r' r := Subrel.relEmbedding r (setOf fun i => ∀ (j : ι), r …
      this : IsWellOrder (↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i))) r'
      ⊢ LE.le (Ordinal.type r) a.cof.ord
    -/
  · rw [← hι, hr]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.refine_2
      a : Ordinal.{u}
      ι : Type u
      f : ι → Ordinal.{u}
      hf : Eq (Ordinal.lsub f) a
      hι : Eq (Cardinal.mk ι) a.cof
      r : ι → ι → Prop
      wo : IsWellOrder ι r
      hr : Eq (Cardinal.mk ι).ord (Ordinal.type r)
      this✝ : IsWellOrder ι r
      r' : ↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i)) → ↑(setOf fun i => …
      hrr' : RelEmbedding r' r := Subrel.relEmbedding r (setOf fun i => ∀ (j : ι), r …
      this : IsWellOrder (↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i))) r'
      i j : Ordinal.{u}
      x✝¹ : LT.lt i (Ordinal.type r')
      h : LT.lt j (Ordinal.type r')
      x✝ : LT.lt i j
      ⊢ r ↑((Ordinal.enum r') ⟨i, x✝¹⟩) ↑((Ordinal.enum r') ⟨j, h⟩)
    -/
  · change r (hrr'.1 _) (hrr'.1 _)
    /-
      case intro.intro.intro.intro.intro.refine_2
      a : Ordinal.{u}
      ι : Type u
      f : ι → Ordinal.{u}
      hf : Eq (Ordinal.lsub f) a
      hι : Eq (Cardinal.mk ι) a.cof
      r : ι → ι → Prop
      wo : IsWellOrder ι r
      hr : Eq (Cardinal.mk ι).ord (Ordinal.type r)
      this✝ : IsWellOrder ι r
      r' : ↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i)) → ↑(setOf fun i => …
      hrr' : RelEmbedding r' r := Subrel.relEmbedding r (setOf fun i => ∀ (j : ι), r …
      this : IsWellOrder (↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i))) r'
      i j : Ordinal.{u}
      x✝¹ : LT.lt i (Ordinal.type r')
      h : LT.lt j (Ordinal.type r')
      x✝ : LT.lt i j
      ⊢ r (hrr'.toEmbedding ((Ordinal.enum r') ⟨i, x✝¹⟩)) (hrr'.toEmbedding ((Ordina …
    -/
    rwa [hrr'.2, @enum_lt_enum _ r']
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.refine_3
      a : Ordinal.{u}
      ι : Type u
      f : ι → Ordinal.{u}
      hf : Eq (Ordinal.lsub f) a
      hι : Eq (Cardinal.mk ι) a.cof
      r : ι → ι → Prop
      wo : IsWellOrder ι r
      hr : Eq (Cardinal.mk ι).ord (Ordinal.type r)
      this✝ : IsWellOrder ι r
      r' : ↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i)) → ↑(setOf fun i => …
      hrr' : RelEmbedding r' r := Subrel.relEmbedding r (setOf fun i => ∀ (j : ι), r …
      this : IsWellOrder (↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i))) r'
      ⊢ LE.le a ((Ordinal.type r').blsub fun i x => f ↑((Ordinal.enum r') ⟨i, x⟩))
    -/
  · rw [← hf, lsub_le_iff]
    /-
      case intro.intro.intro.intro.intro.refine_3
      a : Ordinal.{u}
      ι : Type u
      f : ι → Ordinal.{u}
      hf : Eq (Ordinal.lsub f) a
      hι : Eq (Cardinal.mk ι) a.cof
      r : ι → ι → Prop
      wo : IsWellOrder ι r
      hr : Eq (Cardinal.mk ι).ord (Ordinal.type r)
      this✝ : IsWellOrder ι r
      r' : ↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i)) → ↑(setOf fun i => …
      hrr' : RelEmbedding r' r := Subrel.relEmbedding r (setOf fun i => ∀ (j : ι), r …
      this : IsWellOrder (↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i))) r'
      ⊢ ∀ (i : ι), LT.lt (f i) ((Ordinal.type r').blsub fun i x => f ↑((Ordinal.enum …
    -/
    intro i
    suffices h : ∃ i' hi', f i ≤ bfamilyOfFamily' r' (fun i => f i) i' hi' by
      rcases h with ⟨i', hi', hfg⟩
      exact hfg.trans_lt (lt_blsub _ _ _)
    /-
      case intro.intro.intro.intro.intro.refine_3
      a : Ordinal.{u}
      ι : Type u
      f : ι → Ordinal.{u}
      hf : Eq (Ordinal.lsub f) a
      hι : Eq (Cardinal.mk ι) a.cof
      r : ι → ι → Prop
      wo : IsWellOrder ι r
      hr : Eq (Cardinal.mk ι).ord (Ordinal.type r)
      this✝ : IsWellOrder ι r
      r' : ↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i)) → ↑(setOf fun i => …
      hrr' : RelEmbedding r' r := Subrel.relEmbedding r (setOf fun i => ∀ (j : ι), r …
      this : IsWellOrder (↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i))) r'
      i : ι
      ⊢ Exists fun i' => Exists fun hi' => LE.le (f i) (Ordinal.bfamilyOfFamily' r'  …
    -/
    by_cases h : ∀ j, r j i → f j < f i
      /-
        case pos
        a : Ordinal.{u}
        ι : Type u
        f : ι → Ordinal.{u}
        hf : Eq (Ordinal.lsub f) a
        hι : Eq (Cardinal.mk ι) a.cof
        r : ι → ι → Prop
        wo : IsWellOrder ι r
        hr : Eq (Cardinal.mk ι).ord (Ordinal.type r)
        this✝ : IsWellOrder ι r
        r' : ↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i)) → ↑(setOf fun i => …
        hrr' : RelEmbedding r' r := Subrel.relEmbedding r (setOf fun i => ∀ (j : ι), r …
        this : IsWellOrder (↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i))) r'
        i : ι
        h : ∀ (j : ι), r j i → LT.lt (f j) (f i)
        ⊢ Exists fun i' => Exists fun hi' => LE.le (f i) (Ordinal.bfamilyOfFamily' r'  …
      -/
    · refine ⟨typein r' ⟨i, h⟩, typein_lt_type _ _, ?_⟩
      /-
        case pos
        a : Ordinal.{u}
        ι : Type u
        f : ι → Ordinal.{u}
        hf : Eq (Ordinal.lsub f) a
        hι : Eq (Cardinal.mk ι) a.cof
        r : ι → ι → Prop
        wo : IsWellOrder ι r
        hr : Eq (Cardinal.mk ι).ord (Ordinal.type r)
        this✝ : IsWellOrder ι r
        r' : ↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i)) → ↑(setOf fun i => …
        hrr' : RelEmbedding r' r := Subrel.relEmbedding r (setOf fun i => ∀ (j : ι), r …
        this : IsWellOrder (↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i))) r'
        i : ι
        h : ∀ (j : ι), r j i → LT.lt (f j) (f i)
        ⊢ LE.le (f i) (Ordinal.bfamilyOfFamily' r' (fun i => f ↑i) ((Ordinal.typein r' …
      -/
      rw [bfamilyOfFamily'_typein]
      /-
        🎉 no goals
      -/
      /-
        case neg
        a : Ordinal.{u}
        ι : Type u
        f : ι → Ordinal.{u}
        hf : Eq (Ordinal.lsub f) a
        hι : Eq (Cardinal.mk ι) a.cof
        r : ι → ι → Prop
        wo : IsWellOrder ι r
        hr : Eq (Cardinal.mk ι).ord (Ordinal.type r)
        this✝ : IsWellOrder ι r
        r' : ↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i)) → ↑(setOf fun i => …
        hrr' : RelEmbedding r' r := Subrel.relEmbedding r (setOf fun i => ∀ (j : ι), r …
        this : IsWellOrder (↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i))) r'
        i : ι
        h : Not (∀ (j : ι), r j i → LT.lt (f j) (f i))
        ⊢ Exists fun i' => Exists fun hi' => LE.le (f i) (Ordinal.bfamilyOfFamily' r'  …
      -/
    · push_neg at h
      /-
        case neg
        a : Ordinal.{u}
        ι : Type u
        f : ι → Ordinal.{u}
        hf : Eq (Ordinal.lsub f) a
        hι : Eq (Cardinal.mk ι) a.cof
        r : ι → ι → Prop
        wo : IsWellOrder ι r
        hr : Eq (Cardinal.mk ι).ord (Ordinal.type r)
        this✝ : IsWellOrder ι r
        r' : ↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i)) → ↑(setOf fun i => …
        hrr' : RelEmbedding r' r := Subrel.relEmbedding r (setOf fun i => ∀ (j : ι), r …
        this : IsWellOrder (↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i))) r'
        i : ι
        h : Exists fun j => And (r j i) (LE.le (f i) (f j))
        ⊢ Exists fun i' => Exists fun hi' => LE.le (f i) (Ordinal.bfamilyOfFamily' r'  …
      -/
      cases' wo.wf.min_mem _ h with hji hij
      /-
        case neg.intro
        a : Ordinal.{u}
        ι : Type u
        f : ι → Ordinal.{u}
        hf : Eq (Ordinal.lsub f) a
        hι : Eq (Cardinal.mk ι) a.cof
        r : ι → ι → Prop
        wo : IsWellOrder ι r
        hr : Eq (Cardinal.mk ι).ord (Ordinal.type r)
        this✝ : IsWellOrder ι r
        r' : ↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i)) → ↑(setOf fun i => …
        hrr' : RelEmbedding r' r := Subrel.relEmbedding r (setOf fun i => ∀ (j : ι), r …
        this : IsWellOrder (↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i))) r'
        i : ι
        h : Exists fun j => And (r j i) (LE.le (f i) (f j))
        hji : r (⋯.min (fun x => And (r x i) (LE.le (f i) (f x))) h) i
        hij : LE.le (f i) (f (⋯.min (fun x => And (r x i) (LE.le (f i) (f x))) h))
        ⊢ Exists fun i' => Exists fun hi' => LE.le (f i) (Ordinal.bfamilyOfFamily' r'  …
      -/
      refine ⟨typein r' ⟨_, fun k hkj => lt_of_lt_of_le ?_ hij⟩, typein_lt_type _ _, ?_⟩
        /-
          case neg.intro.refine_1
          a : Ordinal.{u}
          ι : Type u
          f : ι → Ordinal.{u}
          hf : Eq (Ordinal.lsub f) a
          hι : Eq (Cardinal.mk ι) a.cof
          r : ι → ι → Prop
          wo : IsWellOrder ι r
          hr : Eq (Cardinal.mk ι).ord (Ordinal.type r)
          this✝ : IsWellOrder ι r
          r' : ↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i)) → ↑(setOf fun i => …
          hrr' : RelEmbedding r' r := Subrel.relEmbedding r (setOf fun i => ∀ (j : ι), r …
          this : IsWellOrder (↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i))) r'
          i : ι
          h : Exists fun j => And (r j i) (LE.le (f i) (f j))
          hji : r (⋯.min (fun x => And (r x i) (LE.le (f i) (f x))) h) i
          hij : LE.le (f i) (f (⋯.min (fun x => And (r x i) (LE.le (f i) (f x))) h))
          k : ι
          hkj : r k (⋯.min (fun x => And (r x i) (LE.le (f i) (f x))) h)
          ⊢ LT.lt (f k) (f i)
        -/
      · by_contra! H
        /-
          case neg.intro.refine_1
          a : Ordinal.{u}
          ι : Type u
          f : ι → Ordinal.{u}
          hf : Eq (Ordinal.lsub f) a
          hι : Eq (Cardinal.mk ι) a.cof
          r : ι → ι → Prop
          wo : IsWellOrder ι r
          hr : Eq (Cardinal.mk ι).ord (Ordinal.type r)
          this✝ : IsWellOrder ι r
          r' : ↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i)) → ↑(setOf fun i => …
          hrr' : RelEmbedding r' r := Subrel.relEmbedding r (setOf fun i => ∀ (j : ι), r …
          this : IsWellOrder (↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i))) r'
          i : ι
          h : Exists fun j => And (r j i) (LE.le (f i) (f j))
          hji : r (⋯.min (fun x => And (r x i) (LE.le (f i) (f x))) h) i
          hij : LE.le (f i) (f (⋯.min (fun x => And (r x i) (LE.le (f i) (f x))) h))
          k : ι
          hkj : r k (⋯.min (fun x => And (r x i) (LE.le (f i) (f x))) h)
          H : LE.le (f i) (f k)
          ⊢ False
        -/
        exact (wo.wf.not_lt_min _ h ⟨IsTrans.trans _ _ _ hkj hji, H⟩) hkj
        /-
          🎉 no goals
        -/
        /-
          case neg.intro.refine_2
          a : Ordinal.{u}
          ι : Type u
          f : ι → Ordinal.{u}
          hf : Eq (Ordinal.lsub f) a
          hι : Eq (Cardinal.mk ι) a.cof
          r : ι → ι → Prop
          wo : IsWellOrder ι r
          hr : Eq (Cardinal.mk ι).ord (Ordinal.type r)
          this✝ : IsWellOrder ι r
          r' : ↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i)) → ↑(setOf fun i => …
          hrr' : RelEmbedding r' r := Subrel.relEmbedding r (setOf fun i => ∀ (j : ι), r …
          this : IsWellOrder (↑(setOf fun i => ∀ (j : ι), r j i → LT.lt (f j) (f i))) r'
          i : ι
          h : Exists fun j => And (r j i) (LE.le (f i) (f j))
          hji : r (⋯.min (fun x => And (r x i) (LE.le (f i) (f x))) h) i
          hij : LE.le (f i) (f (⋯.min (fun x => And (r x i) (LE.le (f i) (f x))) h))
          ⊢ LE.le (f i) (Ordinal.bfamilyOfFamily' r' (fun i => f ↑i) ((Ordinal.typein r' …
        -/
      · rwa [bfamilyOfFamily'_typein]
        /-
          🎉 no goals
        -/


@[simp]
theorem cof_cof (a : Ordinal.{u}) : cof (cof a).ord = cof a := by
  /-
    a : Ordinal.{u}
    ⊢ Eq a.cof.ord.cof a.cof
  -/
  cases' exists_fundamental_sequence a with f hf
  /-
    case intro
    a : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b a.cof.ord → Ordinal.{u}
    hf : a.IsFundamentalSequence a.cof.ord f
    ⊢ Eq a.cof.ord.cof a.cof
  -/
  cases' exists_fundamental_sequence a.cof.ord with g hg
  /-
    case intro.intro
    a : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b a.cof.ord → Ordinal.{u}
    hf : a.IsFundamentalSequence a.cof.ord f
    g : (b : Ordinal.{u}) → LT.lt b a.cof.ord.cof.ord → Ordinal.{u}
    hg : a.cof.ord.IsFundamentalSequence a.cof.ord.cof.ord g
    ⊢ Eq a.cof.ord.cof a.cof
  -/
  exact ord_injective (hf.trans hg).cof_eq.symm
  /-
    🎉 no goals
  -/


protected theorem IsNormal.isFundamentalSequence {f : Ordinal.{u} → Ordinal.{u}} (hf : IsNormal f)
    {a o} (ha : IsLimit a) {g} (hg : IsFundamentalSequence a o g) :
    IsFundamentalSequence (f a) o fun b hb => f (g b hb) := by
  /-
    f : Ordinal.{u} → Ordinal.{u}
    hf : Ordinal.IsNormal f
    a o : Ordinal.{u}
    ha : a.IsLimit
    g : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
    hg : a.IsFundamentalSequence o g
    ⊢ (f a).IsFundamentalSequence o fun b hb => f (g b hb)
  -/
  refine ⟨?_, @fun i j _ _ h => hf.strictMono (hg.2.1 _ _ h), ?_⟩
    /-
      case refine_1
      f : Ordinal.{u} → Ordinal.{u}
      hf : Ordinal.IsNormal f
      a o : Ordinal.{u}
      ha : a.IsLimit
      g : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
      hg : a.IsFundamentalSequence o g
      ⊢ LE.le o (f a).cof.ord
    -/
  · rcases exists_lsub_cof (f a) with ⟨ι, f', hf', hι⟩
    /-
      case refine_1.intro.intro.intro
      f : Ordinal.{u} → Ordinal.{u}
      hf : Ordinal.IsNormal f
      a o : Ordinal.{u}
      ha : a.IsLimit
      g : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
      hg : a.IsFundamentalSequence o g
      ι : Type u
      f' : ι → Ordinal.{u}
      hf' : Eq (Ordinal.lsub f') (f a)
      hι : Eq (Cardinal.mk ι) (f a).cof
      ⊢ LE.le o (f a).cof.ord
    -/
    rw [← hg.cof_eq, ord_le_ord, ← hι]
    suffices (lsub.{u, u} fun i => sInf { b : Ordinal | f' i ≤ f b }) = a by
      rw [← this]
      apply cof_lsub_le
    have H : ∀ i, ∃ b < a, f' i ≤ f b := fun i => by
      have := lt_lsub.{u, u} f' i
      rw [hf', ← IsNormal.blsub_eq.{u, u} hf ha, lt_blsub_iff] at this
      simpa using this
    /-
      case refine_1.intro.intro.intro
      f : Ordinal.{u} → Ordinal.{u}
      hf : Ordinal.IsNormal f
      a o : Ordinal.{u}
      ha : a.IsLimit
      g : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
      hg : a.IsFundamentalSequence o g
      ι : Type u
      f' : ι → Ordinal.{u}
      hf' : Eq (Ordinal.lsub f') (f a)
      hι : Eq (Cardinal.mk ι) (f a).cof
      H : ∀ (i : ι), Exists fun b => And (LT.lt b a) (LE.le (f' i) (f b))
      ⊢ Eq (Ordinal.lsub fun i => InfSet.sInf (setOf fun b => LE.le (f' i) (f b))) a
    -/
    refine (lsub_le fun i => ?_).antisymm (le_of_forall_lt fun b hb => ?_)
      /-
        case refine_1.intro.intro.intro.refine_1
        f : Ordinal.{u} → Ordinal.{u}
        hf : Ordinal.IsNormal f
        a o : Ordinal.{u}
        ha : a.IsLimit
        g : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
        hg : a.IsFundamentalSequence o g
        ι : Type u
        f' : ι → Ordinal.{u}
        hf' : Eq (Ordinal.lsub f') (f a)
        hι : Eq (Cardinal.mk ι) (f a).cof
        H : ∀ (i : ι), Exists fun b => And (LT.lt b a) (LE.le (f' i) (f b))
        i : ι
        ⊢ LT.lt (InfSet.sInf (setOf fun b => LE.le (f' i) (f b))) a
      -/
    · rcases H i with ⟨b, hb, hb'⟩
      /-
        case refine_1.intro.intro.intro.refine_1.intro.intro
        f : Ordinal.{u} → Ordinal.{u}
        hf : Ordinal.IsNormal f
        a o : Ordinal.{u}
        ha : a.IsLimit
        g : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
        hg : a.IsFundamentalSequence o g
        ι : Type u
        f' : ι → Ordinal.{u}
        hf' : Eq (Ordinal.lsub f') (f a)
        hι : Eq (Cardinal.mk ι) (f a).cof
        H : ∀ (i : ι), Exists fun b => And (LT.lt b a) (LE.le (f' i) (f b))
        i : ι
        b : Ordinal.{u}
        hb : LT.lt b a
        hb' : LE.le (f' i) (f b)
        ⊢ LT.lt (InfSet.sInf (setOf fun b => LE.le (f' i) (f b))) a
      -/
      exact lt_of_le_of_lt (csInf_le' hb') hb
      /-
        🎉 no goals
      -/
      /-
        case refine_1.intro.intro.intro.refine_2
        f : Ordinal.{u} → Ordinal.{u}
        hf : Ordinal.IsNormal f
        a o : Ordinal.{u}
        ha : a.IsLimit
        g : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
        hg : a.IsFundamentalSequence o g
        ι : Type u
        f' : ι → Ordinal.{u}
        hf' : Eq (Ordinal.lsub f') (f a)
        hι : Eq (Cardinal.mk ι) (f a).cof
        H : ∀ (i : ι), Exists fun b => And (LT.lt b a) (LE.le (f' i) (f b))
        b : Ordinal.{u}
        hb : LT.lt b a
        ⊢ LT.lt b (Ordinal.lsub fun i => InfSet.sInf (setOf fun b => LE.le (f' i) (f b …
      -/
    · have := hf.strictMono hb
      /-
        case refine_1.intro.intro.intro.refine_2
        f : Ordinal.{u} → Ordinal.{u}
        hf : Ordinal.IsNormal f
        a o : Ordinal.{u}
        ha : a.IsLimit
        g : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
        hg : a.IsFundamentalSequence o g
        ι : Type u
        f' : ι → Ordinal.{u}
        hf' : Eq (Ordinal.lsub f') (f a)
        hι : Eq (Cardinal.mk ι) (f a).cof
        H : ∀ (i : ι), Exists fun b => And (LT.lt b a) (LE.le (f' i) (f b))
        b : Ordinal.{u}
        hb : LT.lt b a
        this : LT.lt (f b) (f a)
        ⊢ LT.lt b (Ordinal.lsub fun i => InfSet.sInf (setOf fun b => LE.le (f' i) (f b …
      -/
      rw [← hf', lt_lsub_iff] at this
      /-
        case refine_1.intro.intro.intro.refine_2
        f : Ordinal.{u} → Ordinal.{u}
        hf : Ordinal.IsNormal f
        a o : Ordinal.{u}
        ha : a.IsLimit
        g : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
        hg : a.IsFundamentalSequence o g
        ι : Type u
        f' : ι → Ordinal.{u}
        hf' : Eq (Ordinal.lsub f') (f a)
        hι : Eq (Cardinal.mk ι) (f a).cof
        H : ∀ (i : ι), Exists fun b => And (LT.lt b a) (LE.le (f' i) (f b))
        b : Ordinal.{u}
        hb : LT.lt b a
        this : Exists fun i => LE.le (f b) (f' i)
        ⊢ LT.lt b (Ordinal.lsub fun i => InfSet.sInf (setOf fun b => LE.le (f' i) (f b …
      -/
      cases' this with i hi
      /-
        case refine_1.intro.intro.intro.refine_2.intro
        f : Ordinal.{u} → Ordinal.{u}
        hf : Ordinal.IsNormal f
        a o : Ordinal.{u}
        ha : a.IsLimit
        g : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
        hg : a.IsFundamentalSequence o g
        ι : Type u
        f' : ι → Ordinal.{u}
        hf' : Eq (Ordinal.lsub f') (f a)
        hι : Eq (Cardinal.mk ι) (f a).cof
        H : ∀ (i : ι), Exists fun b => And (LT.lt b a) (LE.le (f' i) (f b))
        b : Ordinal.{u}
        hb : LT.lt b a
        i : ι
        hi : LE.le (f b) (f' i)
        ⊢ LT.lt b (Ordinal.lsub fun i => InfSet.sInf (setOf fun b => LE.le (f' i) (f b …
      -/
      rcases H i with ⟨b, _, hb⟩
      exact
        ((le_csInf_iff'' ⟨b, by exact hb⟩).2 fun c hc =>
          hf.strictMono.le_iff_le.1 (hi.trans hc)).trans_lt (lt_lsub _ i)
  · rw [@blsub_comp.{u, u, u} a _ (fun b _ => f b) (@fun i j _ _ h => hf.strictMono.monotone h) g
        hg.2.2]
    /-
      case refine_2
      f : Ordinal.{u} → Ordinal.{u}
      hf : Ordinal.IsNormal f
      a o : Ordinal.{u}
      ha : a.IsLimit
      g : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{u}
      hg : a.IsFundamentalSequence o g
      ⊢ Eq (a.blsub fun b x => f b) (f a)
    -/
    exact IsNormal.blsub_eq.{u, u} hf ha
    /-
      🎉 no goals
    -/


theorem IsNormal.cof_eq {f} (hf : IsNormal f) {a} (ha : IsLimit a) : cof (f a) = cof a :=
  let ⟨_, hg⟩ := exists_fundamental_sequence a
  ord_injective (hf.isFundamentalSequence ha hg).cof_eq


theorem IsNormal.cof_le {f} (hf : IsNormal f) (a) : cof a ≤ cof (f a) := by
  /-
    f : Ordinal.{u_1} → Ordinal.{u_1}
    hf : Ordinal.IsNormal f
    a : Ordinal.{u_1}
    ⊢ LE.le a.cof (f a).cof
  -/
  rcases zero_or_succ_or_limit a with (rfl | ⟨b, rfl⟩ | ha)
    /-
      case inl
      f : Ordinal.{u_1} → Ordinal.{u_1}
      hf : Ordinal.IsNormal f
      ⊢ LE.le (Ordinal.cof 0) (f 0).cof
    -/
  · rw [cof_zero]
    /-
      case inl
      f : Ordinal.{u_1} → Ordinal.{u_1}
      hf : Ordinal.IsNormal f
      ⊢ LE.le 0 (f 0).cof
    -/
    exact zero_le _
    /-
      🎉 no goals
    -/
    /-
      case inr.inl.intro
      f : Ordinal.{u_1} → Ordinal.{u_1}
      hf : Ordinal.IsNormal f
      b : Ordinal.{u_1}
      ⊢ LE.le (Order.succ b).cof (f (Order.succ b)).cof
    -/
  · rw [cof_succ, Cardinal.one_le_iff_ne_zero, cof_ne_zero, ← Ordinal.pos_iff_ne_zero]
    /-
      case inr.inl.intro
      f : Ordinal.{u_1} → Ordinal.{u_1}
      hf : Ordinal.IsNormal f
      b : Ordinal.{u_1}
      ⊢ LT.lt 0 (f (Order.succ b))
    -/
    exact (Ordinal.zero_le (f b)).trans_lt (hf.1 b)
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      f : Ordinal.{u_1} → Ordinal.{u_1}
      hf : Ordinal.IsNormal f
      a : Ordinal.{u_1}
      ha : a.IsLimit
      ⊢ LE.le a.cof (f a).cof
    -/
  · rw [hf.cof_eq ha]
    /-
      🎉 no goals
    -/


@[simp]
theorem cof_add (a b : Ordinal) : b ≠ 0 → cof (a + b) = cof b := fun h => by
  /-
    a b : Ordinal.{u_1}
    h : Ne b 0
    ⊢ Eq (HAdd.hAdd a b).cof b.cof
  -/
  rcases zero_or_succ_or_limit b with (rfl | ⟨c, rfl⟩ | hb)
    /-
      case inl
      a : Ordinal.{u_1}
      h : Ne 0 0
      ⊢ Eq (HAdd.hAdd a 0).cof (Ordinal.cof 0)
    -/
  · contradiction
    /-
      🎉 no goals
    -/
    /-
      case inr.inl.intro
      a c : Ordinal.{u_1}
      h : Ne (Order.succ c) 0
      ⊢ Eq (HAdd.hAdd a (Order.succ c)).cof (Order.succ c).cof
    -/
  · rw [add_succ, cof_succ, cof_succ]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      a b : Ordinal.{u_1}
      h : Ne b 0
      hb : b.IsLimit
      ⊢ Eq (HAdd.hAdd a b).cof b.cof
    -/
  · exact (isNormal_add_right a).cof_eq hb
    /-
      🎉 no goals
    -/


theorem aleph0_le_cof {o} : ℵ₀ ≤ cof o ↔ IsLimit o := by
  /-
    o : Ordinal.{u_1}
    ⊢ Iff (LE.le Cardinal.aleph0 o.cof) o.IsLimit
  -/
  rcases zero_or_succ_or_limit o with (rfl | ⟨o, rfl⟩ | l)
    /-
      case inl
      ⊢ Iff (LE.le Cardinal.aleph0 (Ordinal.cof 0)) (Ordinal.IsLimit 0)
    -/
  · simp [not_zero_isLimit, Cardinal.aleph0_ne_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl.intro
      o : Ordinal.{u_1}
      ⊢ Iff (LE.le Cardinal.aleph0 (Order.succ o).cof) (Order.succ o).IsLimit
    -/
  · simp [not_succ_isLimit, Cardinal.one_lt_aleph0]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      o : Ordinal.{u_1}
      l : o.IsLimit
      ⊢ Iff (LE.le Cardinal.aleph0 o.cof) o.IsLimit
    -/
  · simp only [l, iff_true]
    /-
      case inr.inr
      o : Ordinal.{u_1}
      l : o.IsLimit
      ⊢ LE.le Cardinal.aleph0 o.cof
    -/
    refine le_of_not_lt fun h => ?_
    /-
      case inr.inr
      o : Ordinal.{u_1}
      l : o.IsLimit
      h : LT.lt o.cof Cardinal.aleph0
      ⊢ False
    -/
    cases' Cardinal.lt_aleph0.1 h with n e
    /-
      case inr.inr.intro
      o : Ordinal.{u_1}
      l : o.IsLimit
      h : LT.lt o.cof Cardinal.aleph0
      n : Nat
      e : Eq o.cof ↑n
      ⊢ False
    -/
    have := cof_cof o
    /-
      case inr.inr.intro
      o : Ordinal.{u_1}
      l : o.IsLimit
      h : LT.lt o.cof Cardinal.aleph0
      n : Nat
      e : Eq o.cof ↑n
      this : Eq o.cof.ord.cof o.cof
      ⊢ False
    -/
    rw [e, ord_nat] at this
    /-
      case inr.inr.intro
      o : Ordinal.{u_1}
      l : o.IsLimit
      h : LT.lt o.cof Cardinal.aleph0
      n : Nat
      e : Eq o.cof ↑n
      this : Eq (↑n).cof ↑n
      ⊢ False
    -/
    cases n
      /-
        case inr.inr.intro.zero
        o : Ordinal.{u_1}
        l : o.IsLimit
        h : LT.lt o.cof Cardinal.aleph0
        e : Eq o.cof ↑0
        this : Eq (↑0).cof ↑0
        ⊢ False
      -/
    · simp at e
      /-
        case inr.inr.intro.zero
        o : Ordinal.{u_1}
        l : o.IsLimit
        h : LT.lt o.cof Cardinal.aleph0
        this : Eq (↑0).cof ↑0
        e : Eq o 0
        ⊢ False
      -/
      simp [e, not_zero_isLimit] at l
      /-
        🎉 no goals
      -/
      /-
        case inr.inr.intro.succ
        o : Ordinal.{u_1}
        l : o.IsLimit
        h : LT.lt o.cof Cardinal.aleph0
        n✝ : Nat
        e : Eq o.cof ↑(HAdd.hAdd n✝ 1)
        this : Eq (↑(HAdd.hAdd n✝ 1)).cof ↑(HAdd.hAdd n✝ 1)
        ⊢ False
      -/
    · rw [natCast_succ, cof_succ] at this
      /-
        case inr.inr.intro.succ
        o : Ordinal.{u_1}
        l : o.IsLimit
        h : LT.lt o.cof Cardinal.aleph0
        n✝ : Nat
        e : Eq o.cof ↑(HAdd.hAdd n✝ 1)
        this : Eq 1 ↑(HAdd.hAdd n✝ 1)
        ⊢ False
      -/
      rw [← this, cof_eq_one_iff_is_succ] at e
      /-
        case inr.inr.intro.succ
        o : Ordinal.{u_1}
        l : o.IsLimit
        h : LT.lt o.cof Cardinal.aleph0
        n✝ : Nat
        e : Exists fun a => Eq o (Order.succ a)
        this : Eq 1 ↑(HAdd.hAdd n✝ 1)
        ⊢ False
      -/
      rcases e with ⟨a, rfl⟩
      /-
        case inr.inr.intro.succ.intro
        n✝ : Nat
        this : Eq 1 ↑(HAdd.hAdd n✝ 1)
        a : Ordinal.{u_1}
        l : (Order.succ a).IsLimit
        h : LT.lt (Order.succ a).cof Cardinal.aleph0
        ⊢ False
      -/
      exact not_succ_isLimit _ l
      /-
        🎉 no goals
      -/


@[simp]
theorem cof_preOmega {o : Ordinal} (ho : IsSuccPrelimit o) : (preOmega o).cof = o.cof := by
  /-
    o : Ordinal.{u_1}
    ho : Order.IsSuccPrelimit o
    ⊢ Eq (Ordinal.preOmega o).cof o.cof
  -/
  by_cases h : IsMin o
    /-
      case pos
      o : Ordinal.{u_1}
      ho : Order.IsSuccPrelimit o
      h : IsMin o
      ⊢ Eq (Ordinal.preOmega o).cof o.cof
    -/
  · simp [h.eq_bot]
    /-
      🎉 no goals
    -/
    /-
      case neg
      o : Ordinal.{u_1}
      ho : Order.IsSuccPrelimit o
      h : Not (IsMin o)
      ⊢ Eq (Ordinal.preOmega o).cof o.cof
    -/
  · exact isNormal_preOmega.cof_eq ⟨h, ho⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem cof_omega {o : Ordinal} (ho : o.IsLimit) : (ω_ o).cof = o.cof :=
  isNormal_omega.cof_eq ho


set_option linter.deprecated false in
@[deprecated cof_preOmega (since := "2024-10-22")]
theorem preAleph_cof {o : Ordinal} (ho : o.IsLimit) : (preAleph o).ord.cof = o.cof :=
  aleph'_isNormal.cof_eq ho


set_option linter.deprecated false in
@[deprecated cof_preOmega (since := "2024-10-22")]
theorem aleph'_cof {o : Ordinal} (ho : o.IsLimit) : (aleph' o).ord.cof = o.cof :=
  aleph'_isNormal.cof_eq ho


set_option linter.deprecated false in
@[deprecated cof_omega (since := "2024-10-22")]
theorem aleph_cof {o : Ordinal} (ho : o.IsLimit) : (ℵ_  o).ord.cof = o.cof :=
  aleph_isNormal.cof_eq ho


@[simp]
theorem cof_omega0 : cof ω = ℵ₀ :=
  (aleph0_le_cof.2 isLimit_omega0).antisymm' <| by
    /-
      ⊢ LE.le Ordinal.omega0.cof Cardinal.aleph0
    -/
    rw [← card_omega0]
    /-
      ⊢ LE.le Ordinal.omega0.cof Ordinal.omega0.card
    -/
    apply cof_le_card
    /-
      🎉 no goals
    -/


theorem cof_eq' (r : α → α → Prop) [IsWellOrder α r] (h : IsLimit (type r)) :
    ∃ S : Set α, (∀ a, ∃ b ∈ S, r a b) ∧ #S = cof (type r) :=
  let ⟨S, H, e⟩ := cof_eq r
  ⟨S, fun a =>
    let a' := enum r ⟨_, h.succ_lt (typein_lt_type r a)⟩
    let ⟨b, h, ab⟩ := H a'
    ⟨b, h,
      (IsOrderConnected.conn a b a' <|
            (typein_lt_typein r).1
              (by
                /-
                  α : Type u
                  r : α → α → Prop
                  inst✝ : IsWellOrder α r
                  h✝ : (Ordinal.type r).IsLimit
                  S : Set α
                  H : Set.Unbounded r S
                  e : Eq (Cardinal.mk ↑S) (Ordinal.type r).cof
                  a : α
                  a' : α := (Ordinal.enum r) ⟨Order.succ ((Ordinal.typein r).toRelEmbedding a), ⋯⟩
                  b : α
                  h : Membership.mem S b
                  ab : Not (r b a')
                  ⊢ LT.lt ((Ordinal.typein r).toRelEmbedding a) ((Ordinal.typein r).toRelEmbeddi …
                -/
                rw [typein_enum]
                /-
                  α : Type u
                  r : α → α → Prop
                  inst✝ : IsWellOrder α r
                  h✝ : (Ordinal.type r).IsLimit
                  S : Set α
                  H : Set.Unbounded r S
                  e : Eq (Cardinal.mk ↑S) (Ordinal.type r).cof
                  a : α
                  a' : α := (Ordinal.enum r) ⟨Order.succ ((Ordinal.typein r).toRelEmbedding a), ⋯⟩
                  b : α
                  h : Membership.mem S b
                  ab : Not (r b a')
                  ⊢ LT.lt ((Ordinal.typein r).toRelEmbedding a) (Order.succ ((Ordinal.typein r). …
                -/
                exact lt_succ (typein _ _))).resolve_right
                /-
                  🎉 no goals
                -/
        ab⟩,
    e⟩


@[simp]
theorem cof_univ : cof univ.{u, v} = Cardinal.univ.{u, v} :=
  le_antisymm (cof_le_card _)
    (by
      /-
        ⊢ LE.le Cardinal.univ.{u, v} Ordinal.univ.{u, v}.cof
      -/
      refine le_of_forall_lt fun c h => ?_
      /-
        c : Cardinal.{max (u + 1) v}
        h : LT.lt c Cardinal.univ.{u, v}
        ⊢ LT.lt c Ordinal.univ.{u, v}.cof
      -/
      rcases lt_univ'.1 h with ⟨c, rfl⟩
      /-
        case intro
        c : Cardinal.{u}
        h : LT.lt (Cardinal.lift.{max (u + 1) v, u} c) Cardinal.univ.{u, v}
        ⊢ LT.lt (Cardinal.lift.{max (u + 1) v, u} c) Ordinal.univ.{u, v}.cof
      -/
      rcases @cof_eq Ordinal.{u} (· < ·) _ with ⟨S, H, Se⟩
      /-
        case intro.intro.intro
        c : Cardinal.{u}
        h : LT.lt (Cardinal.lift.{max (u + 1) v, u} c) Cardinal.univ.{u, v}
        S : Set Ordinal.{u}
        H : Set.Unbounded (fun x1 x2 => LT.lt x1 x2) S
        Se : Eq (Cardinal.mk ↑S) (Ordinal.type fun x1 x2 => LT.lt x1 x2).cof
        ⊢ LT.lt (Cardinal.lift.{max (u + 1) v, u} c) Ordinal.univ.{u, v}.cof
      -/
      rw [univ, ← lift_cof, ← Cardinal.lift_lift.{u+1, v, u}, Cardinal.lift_lt, ← Se]
      /-
        case intro.intro.intro
        c : Cardinal.{u}
        h : LT.lt (Cardinal.lift.{max (u + 1) v, u} c) Cardinal.univ.{u, v}
        S : Set Ordinal.{u}
        H : Set.Unbounded (fun x1 x2 => LT.lt x1 x2) S
        Se : Eq (Cardinal.mk ↑S) (Ordinal.type fun x1 x2 => LT.lt x1 x2).cof
        ⊢ LT.lt (Cardinal.lift.{u + 1, u} c) (Cardinal.mk ↑S)
      -/
      refine lt_of_not_ge fun h => ?_
      /-
        case intro.intro.intro
        c : Cardinal.{u}
        h✝ : LT.lt (Cardinal.lift.{max (u + 1) v, u} c) Cardinal.univ.{u, v}
        S : Set Ordinal.{u}
        H : Set.Unbounded (fun x1 x2 => LT.lt x1 x2) S
        Se : Eq (Cardinal.mk ↑S) (Ordinal.type fun x1 x2 => LT.lt x1 x2).cof
        h : GE.ge (Cardinal.lift.{u + 1, u} c) (Cardinal.mk ↑S)
        ⊢ False
      -/
      cases' Cardinal.mem_range_lift_of_le h with a e
      /-
        case intro.intro.intro.intro
        c : Cardinal.{u}
        h✝ : LT.lt (Cardinal.lift.{max (u + 1) v, u} c) Cardinal.univ.{u, v}
        S : Set Ordinal.{u}
        H : Set.Unbounded (fun x1 x2 => LT.lt x1 x2) S
        Se : Eq (Cardinal.mk ↑S) (Ordinal.type fun x1 x2 => LT.lt x1 x2).cof
        h : GE.ge (Cardinal.lift.{u + 1, u} c) (Cardinal.mk ↑S)
        a : Cardinal.{u}
        e : Eq (Cardinal.lift.{u + 1, u} a) (Cardinal.mk ↑S)
        ⊢ False
      -/
      refine Quotient.inductionOn a (fun α e => ?_) e
      /-
        case intro.intro.intro.intro
        c : Cardinal.{u}
        h✝ : LT.lt (Cardinal.lift.{max (u + 1) v, u} c) Cardinal.univ.{u, v}
        S : Set Ordinal.{u}
        H : Set.Unbounded (fun x1 x2 => LT.lt x1 x2) S
        Se : Eq (Cardinal.mk ↑S) (Ordinal.type fun x1 x2 => LT.lt x1 x2).cof
        h : GE.ge (Cardinal.lift.{u + 1, u} c) (Cardinal.mk ↑S)
        a : Cardinal.{u}
        e✝ : Eq (Cardinal.lift.{u + 1, u} a) (Cardinal.mk ↑S)
        α : Type u
        e : Eq (Cardinal.lift.{u + 1, u} (Quotient.mk Cardinal.isEquivalent α)) (Cardi …
        ⊢ False
      -/
      cases' Quotient.exact e with f
      /-
        case intro.intro.intro.intro.intro
        c : Cardinal.{u}
        h✝ : LT.lt (Cardinal.lift.{max (u + 1) v, u} c) Cardinal.univ.{u, v}
        S : Set Ordinal.{u}
        H : Set.Unbounded (fun x1 x2 => LT.lt x1 x2) S
        Se : Eq (Cardinal.mk ↑S) (Ordinal.type fun x1 x2 => LT.lt x1 x2).cof
        h : GE.ge (Cardinal.lift.{u + 1, u} c) (Cardinal.mk ↑S)
        a : Cardinal.{u}
        e✝ : Eq (Cardinal.lift.{u + 1, u} a) (Cardinal.mk ↑S)
        α : Type u
        e : Eq (Cardinal.lift.{u + 1, u} (Quotient.mk Cardinal.isEquivalent α)) (Cardi …
        f : Equiv (ULift.{u + 1, u} α) ↑S
        ⊢ False
      -/
      have f := Equiv.ulift.symm.trans f
      /-
        case intro.intro.intro.intro.intro
        c : Cardinal.{u}
        h✝ : LT.lt (Cardinal.lift.{max (u + 1) v, u} c) Cardinal.univ.{u, v}
        S : Set Ordinal.{u}
        H : Set.Unbounded (fun x1 x2 => LT.lt x1 x2) S
        Se : Eq (Cardinal.mk ↑S) (Ordinal.type fun x1 x2 => LT.lt x1 x2).cof
        h : GE.ge (Cardinal.lift.{u + 1, u} c) (Cardinal.mk ↑S)
        a : Cardinal.{u}
        e✝ : Eq (Cardinal.lift.{u + 1, u} a) (Cardinal.mk ↑S)
        α : Type u
        e : Eq (Cardinal.lift.{u + 1, u} (Quotient.mk Cardinal.isEquivalent α)) (Cardi …
        f✝ : Equiv (ULift.{u + 1, u} α) ↑S
        f : Equiv α ↑S
        ⊢ False
      -/
      let g a := (f a).1
      /-
        case intro.intro.intro.intro.intro
        c : Cardinal.{u}
        h✝ : LT.lt (Cardinal.lift.{max (u + 1) v, u} c) Cardinal.univ.{u, v}
        S : Set Ordinal.{u}
        H : Set.Unbounded (fun x1 x2 => LT.lt x1 x2) S
        Se : Eq (Cardinal.mk ↑S) (Ordinal.type fun x1 x2 => LT.lt x1 x2).cof
        h : GE.ge (Cardinal.lift.{u + 1, u} c) (Cardinal.mk ↑S)
        a : Cardinal.{u}
        e✝ : Eq (Cardinal.lift.{u + 1, u} a) (Cardinal.mk ↑S)
        α : Type u
        e : Eq (Cardinal.lift.{u + 1, u} (Quotient.mk Cardinal.isEquivalent α)) (Cardi …
        f✝ : Equiv (ULift.{u + 1, u} α) ↑S
        f : Equiv α ↑S
        g : α → Ordinal.{u} := fun a => ↑(f a)
        ⊢ False
      -/
      let o := succ (iSup g)
      /-
        case intro.intro.intro.intro.intro
        c : Cardinal.{u}
        h✝ : LT.lt (Cardinal.lift.{max (u + 1) v, u} c) Cardinal.univ.{u, v}
        S : Set Ordinal.{u}
        H : Set.Unbounded (fun x1 x2 => LT.lt x1 x2) S
        Se : Eq (Cardinal.mk ↑S) (Ordinal.type fun x1 x2 => LT.lt x1 x2).cof
        h : GE.ge (Cardinal.lift.{u + 1, u} c) (Cardinal.mk ↑S)
        a : Cardinal.{u}
        e✝ : Eq (Cardinal.lift.{u + 1, u} a) (Cardinal.mk ↑S)
        α : Type u
        e : Eq (Cardinal.lift.{u + 1, u} (Quotient.mk Cardinal.isEquivalent α)) (Cardi …
        f✝ : Equiv (ULift.{u + 1, u} α) ↑S
        f : Equiv α ↑S
        g : α → Ordinal.{u} := fun a => ↑(f a)
        o : Ordinal.{u} := Order.succ (iSup g)
        ⊢ False
      -/
      rcases H o with ⟨b, h, l⟩
      /-
        case intro.intro.intro.intro.intro.intro.intro
        c : Cardinal.{u}
        h✝¹ : LT.lt (Cardinal.lift.{max (u + 1) v, u} c) Cardinal.univ.{u, v}
        S : Set Ordinal.{u}
        H : Set.Unbounded (fun x1 x2 => LT.lt x1 x2) S
        Se : Eq (Cardinal.mk ↑S) (Ordinal.type fun x1 x2 => LT.lt x1 x2).cof
        h✝ : GE.ge (Cardinal.lift.{u + 1, u} c) (Cardinal.mk ↑S)
        a : Cardinal.{u}
        e✝ : Eq (Cardinal.lift.{u + 1, u} a) (Cardinal.mk ↑S)
        α : Type u
        e : Eq (Cardinal.lift.{u + 1, u} (Quotient.mk Cardinal.isEquivalent α)) (Cardi …
        f✝ : Equiv (ULift.{u + 1, u} α) ↑S
        f : Equiv α ↑S
        g : α → Ordinal.{u} := fun a => ↑(f a)
        o : Ordinal.{u} := Order.succ (iSup g)
        b : Ordinal.{u}
        h : Membership.mem S b
        l : Not ((fun x1 x2 => LT.lt x1 x2) b o)
        ⊢ False
      -/
      refine l (lt_succ_iff.2 ?_)
      /-
        case intro.intro.intro.intro.intro.intro.intro
        c : Cardinal.{u}
        h✝¹ : LT.lt (Cardinal.lift.{max (u + 1) v, u} c) Cardinal.univ.{u, v}
        S : Set Ordinal.{u}
        H : Set.Unbounded (fun x1 x2 => LT.lt x1 x2) S
        Se : Eq (Cardinal.mk ↑S) (Ordinal.type fun x1 x2 => LT.lt x1 x2).cof
        h✝ : GE.ge (Cardinal.lift.{u + 1, u} c) (Cardinal.mk ↑S)
        a : Cardinal.{u}
        e✝ : Eq (Cardinal.lift.{u + 1, u} a) (Cardinal.mk ↑S)
        α : Type u
        e : Eq (Cardinal.lift.{u + 1, u} (Quotient.mk Cardinal.isEquivalent α)) (Cardi …
        f✝ : Equiv (ULift.{u + 1, u} α) ↑S
        f : Equiv α ↑S
        g : α → Ordinal.{u} := fun a => ↑(f a)
        o : Ordinal.{u} := Order.succ (iSup g)
        b : Ordinal.{u}
        h : Membership.mem S b
        l : Not ((fun x1 x2 => LT.lt x1 x2) b o)
        ⊢ LE.le b (iSup g)
      -/
      rw [← show g (f.symm ⟨b, h⟩) = b by simp [g]]
      /-
        case intro.intro.intro.intro.intro.intro.intro
        c : Cardinal.{u}
        h✝¹ : LT.lt (Cardinal.lift.{max (u + 1) v, u} c) Cardinal.univ.{u, v}
        S : Set Ordinal.{u}
        H : Set.Unbounded (fun x1 x2 => LT.lt x1 x2) S
        Se : Eq (Cardinal.mk ↑S) (Ordinal.type fun x1 x2 => LT.lt x1 x2).cof
        h✝ : GE.ge (Cardinal.lift.{u + 1, u} c) (Cardinal.mk ↑S)
        a : Cardinal.{u}
        e✝ : Eq (Cardinal.lift.{u + 1, u} a) (Cardinal.mk ↑S)
        α : Type u
        e : Eq (Cardinal.lift.{u + 1, u} (Quotient.mk Cardinal.isEquivalent α)) (Cardi …
        f✝ : Equiv (ULift.{u + 1, u} α) ↑S
        f : Equiv α ↑S
        g : α → Ordinal.{u} := fun a => ↑(f a)
        o : Ordinal.{u} := Order.succ (iSup g)
        b : Ordinal.{u}
        h : Membership.mem S b
        l : Not ((fun x1 x2 => LT.lt x1 x2) b o)
        ⊢ LE.le (g (f.symm ⟨b, h⟩)) (iSup g)
      -/
      apply Ordinal.le_iSup)
      /-
        🎉 no goals
      -/


/-- If the union of s is unbounded and s is smaller than the cofinality,
  then s has an unbounded member -/
theorem unbounded_of_unbounded_sUnion (r : α → α → Prop) [wo : IsWellOrder α r] {s : Set (Set α)}
    (h₁ : Unbounded r <| ⋃₀ s) (h₂ : #s < Order.cof (swap rᶜ)) : ∃ x ∈ s, Unbounded r x := by
  /-
    α : Type u
    r : α → α → Prop
    wo : IsWellOrder α r
    s : Set (Set α)
    h₁ : Set.Unbounded r s.sUnion
    h₂ : LT.lt (Cardinal.mk ↑s) (Order.cof (Function.swap (HasCompl.compl r)))
    ⊢ Exists fun x => And (Membership.mem s x) (Set.Unbounded r x)
  -/
  by_contra! h
  /-
    α : Type u
    r : α → α → Prop
    wo : IsWellOrder α r
    s : Set (Set α)
    h₁ : Set.Unbounded r s.sUnion
    h₂ : LT.lt (Cardinal.mk ↑s) (Order.cof (Function.swap (HasCompl.compl r)))
    h : ∀ (x : Set α), Membership.mem s x → Not (Set.Unbounded r x)
    ⊢ False
  -/
  simp_rw [not_unbounded_iff] at h
  /-
    α : Type u
    r : α → α → Prop
    wo : IsWellOrder α r
    s : Set (Set α)
    h₁ : Set.Unbounded r s.sUnion
    h₂ : LT.lt (Cardinal.mk ↑s) (Order.cof (Function.swap (HasCompl.compl r)))
    h : ∀ (x : Set α), Membership.mem s x → Set.Bounded r x
    ⊢ False
  -/
  let f : s → α := fun x : s => wo.wf.sup x (h x.1 x.2)
  /-
    α : Type u
    r : α → α → Prop
    wo : IsWellOrder α r
    s : Set (Set α)
    h₁ : Set.Unbounded r s.sUnion
    h₂ : LT.lt (Cardinal.mk ↑s) (Order.cof (Function.swap (HasCompl.compl r)))
    h : ∀ (x : Set α), Membership.mem s x → Set.Bounded r x
    f : ↑s → α := fun x => ⋯.sup ↑x ⋯
    ⊢ False
  -/
  refine h₂.not_le (le_trans (csInf_le' ⟨range f, fun x => ?_, rfl⟩) mk_range_le)
  /-
    α : Type u
    r : α → α → Prop
    wo : IsWellOrder α r
    s : Set (Set α)
    h₁ : Set.Unbounded r s.sUnion
    h₂ : LT.lt (Cardinal.mk ↑s) (Order.cof (Function.swap (HasCompl.compl r)))
    h : ∀ (x : Set α), Membership.mem s x → Set.Bounded r x
    f : ↑s → α := fun x => ⋯.sup ↑x ⋯
    x : α
    ⊢ Exists fun b => And (Membership.mem (Set.range f) b) (Function.swap (HasComp …
  -/
  rcases h₁ x with ⟨y, ⟨c, hc, hy⟩, hxy⟩
  /-
    case intro.intro.intro.intro
    α : Type u
    r : α → α → Prop
    wo : IsWellOrder α r
    s : Set (Set α)
    h₁ : Set.Unbounded r s.sUnion
    h₂ : LT.lt (Cardinal.mk ↑s) (Order.cof (Function.swap (HasCompl.compl r)))
    h : ∀ (x : Set α), Membership.mem s x → Set.Bounded r x
    f : ↑s → α := fun x => ⋯.sup ↑x ⋯
    x y : α
    hxy : Not (r y x)
    c : Set α
    hc : Membership.mem s c
    hy : Membership.mem c y
    ⊢ Exists fun b => And (Membership.mem (Set.range f) b) (Function.swap (HasComp …
  -/
  exact ⟨f ⟨c, hc⟩, mem_range_self _, fun hxz => hxy (Trans.trans (wo.wf.lt_sup _ hy) hxz)⟩
  /-
    🎉 no goals
  -/


/-- If the union of s is unbounded and s is smaller than the cofinality,
  then s has an unbounded member -/
theorem unbounded_of_unbounded_iUnion {α β : Type u} (r : α → α → Prop) [wo : IsWellOrder α r]
    (s : β → Set α) (h₁ : Unbounded r <| ⋃ x, s x) (h₂ : #β < Order.cof (swap rᶜ)) :
    ∃ x : β, Unbounded r (s x) := by
  /-
    α β : Type u
    r : α → α → Prop
    wo : IsWellOrder α r
    s : β → Set α
    h₁ : Set.Unbounded r (Set.iUnion fun x => s x)
    h₂ : LT.lt (Cardinal.mk β) (Order.cof (Function.swap (HasCompl.compl r)))
    ⊢ Exists fun x => Set.Unbounded r (s x)
  -/
  rw [← sUnion_range] at h₁
  /-
    α β : Type u
    r : α → α → Prop
    wo : IsWellOrder α r
    s : β → Set α
    h₁ : Set.Unbounded r (Set.range s).sUnion
    h₂ : LT.lt (Cardinal.mk β) (Order.cof (Function.swap (HasCompl.compl r)))
    ⊢ Exists fun x => Set.Unbounded r (s x)
  -/
  rcases unbounded_of_unbounded_sUnion r h₁ (mk_range_le.trans_lt h₂) with ⟨_, ⟨x, rfl⟩, u⟩
  /-
    case intro.intro.intro
    α β : Type u
    r : α → α → Prop
    wo : IsWellOrder α r
    s : β → Set α
    h₁ : Set.Unbounded r (Set.range s).sUnion
    h₂ : LT.lt (Cardinal.mk β) (Order.cof (Function.swap (HasCompl.compl r)))
    x : β
    u : Set.Unbounded r (s x)
    ⊢ Exists fun x => Set.Unbounded r (s x)
  -/
  exact ⟨x, u⟩
  /-
    🎉 no goals
  -/


/-- The infinite pigeonhole principle -/
theorem infinite_pigeonhole {β α : Type u} (f : β → α) (h₁ : ℵ₀ ≤ #β) (h₂ : #α < (#β).ord.cof) :
    ∃ a : α, #(f ⁻¹' {a}) = #β := by
  have : ∃ a, #β ≤ #(f ⁻¹' {a}) := by
    by_contra! h
    apply mk_univ.not_lt
    rw [← preimage_univ, ← iUnion_of_singleton, preimage_iUnion]
    exact
      mk_iUnion_le_sum_mk.trans_lt
        ((sum_le_iSup _).trans_lt <| mul_lt_of_lt h₁ (h₂.trans_le <| cof_ord_le _) (iSup_lt h₂ h))
  /-
    β α : Type u
    f : β → α
    h₁ : LE.le Cardinal.aleph0 (Cardinal.mk β)
    h₂ : LT.lt (Cardinal.mk α) (Cardinal.mk β).ord.cof
    this : Exists fun a => LE.le (Cardinal.mk β) (Cardinal.mk ↑(Set.preimage f (Si …
    ⊢ Exists fun a => Eq (Cardinal.mk ↑(Set.preimage f (Singleton.singleton a))) ( …
  -/
  cases' this with x h
  /-
    case intro
    β α : Type u
    f : β → α
    h₁ : LE.le Cardinal.aleph0 (Cardinal.mk β)
    h₂ : LT.lt (Cardinal.mk α) (Cardinal.mk β).ord.cof
    x : α
    h : LE.le (Cardinal.mk β) (Cardinal.mk ↑(Set.preimage f (Singleton.singleton x …
    ⊢ Exists fun a => Eq (Cardinal.mk ↑(Set.preimage f (Singleton.singleton a))) ( …
  -/
  refine ⟨x, h.antisymm' ?_⟩
  /-
    case intro
    β α : Type u
    f : β → α
    h₁ : LE.le Cardinal.aleph0 (Cardinal.mk β)
    h₂ : LT.lt (Cardinal.mk α) (Cardinal.mk β).ord.cof
    x : α
    h : LE.le (Cardinal.mk β) (Cardinal.mk ↑(Set.preimage f (Singleton.singleton x …
    ⊢ LE.le (Cardinal.mk ↑(Set.preimage f (Singleton.singleton x))) (Cardinal.mk β)
  -/
  rw [le_mk_iff_exists_set]
  /-
    case intro
    β α : Type u
    f : β → α
    h₁ : LE.le Cardinal.aleph0 (Cardinal.mk β)
    h₂ : LT.lt (Cardinal.mk α) (Cardinal.mk β).ord.cof
    x : α
    h : LE.le (Cardinal.mk β) (Cardinal.mk ↑(Set.preimage f (Singleton.singleton x …
    ⊢ Exists fun p => Eq (Cardinal.mk ↑p) (Cardinal.mk ↑(Set.preimage f (Singleton …
  -/
  exact ⟨_, rfl⟩
  /-
    🎉 no goals
  -/


/-- Pigeonhole principle for a cardinality below the cardinality of the domain -/
theorem infinite_pigeonhole_card {β α : Type u} (f : β → α) (θ : Cardinal) (hθ : θ ≤ #β)
    (h₁ : ℵ₀ ≤ θ) (h₂ : #α < θ.ord.cof) : ∃ a : α, θ ≤ #(f ⁻¹' {a}) := by
  /-
    β α : Type u
    f : β → α
    θ : Cardinal.{u}
    hθ : LE.le θ (Cardinal.mk β)
    h₁ : LE.le Cardinal.aleph0 θ
    h₂ : LT.lt (Cardinal.mk α) θ.ord.cof
    ⊢ Exists fun a => LE.le θ (Cardinal.mk ↑(Set.preimage f (Singleton.singleton a …
  -/
  rcases le_mk_iff_exists_set.1 hθ with ⟨s, rfl⟩
  /-
    case intro
    β α : Type u
    f : β → α
    s : Set β
    hθ : LE.le (Cardinal.mk ↑s) (Cardinal.mk β)
    h₁ : LE.le Cardinal.aleph0 (Cardinal.mk ↑s)
    h₂ : LT.lt (Cardinal.mk α) (Cardinal.mk ↑s).ord.cof
    ⊢ Exists fun a => LE.le (Cardinal.mk ↑s) (Cardinal.mk ↑(Set.preimage f (Single …
  -/
  cases' infinite_pigeonhole (f ∘ Subtype.val : s → α) h₁ h₂ with a ha
  /-
    case intro.intro
    β α : Type u
    f : β → α
    s : Set β
    hθ : LE.le (Cardinal.mk ↑s) (Cardinal.mk β)
    h₁ : LE.le Cardinal.aleph0 (Cardinal.mk ↑s)
    h₂ : LT.lt (Cardinal.mk α) (Cardinal.mk ↑s).ord.cof
    a : α
    ha : Eq (Cardinal.mk ↑(Set.preimage (Function.comp f Subtype.val) (Singleton.s …
    ⊢ Exists fun a => LE.le (Cardinal.mk ↑s) (Cardinal.mk ↑(Set.preimage f (Single …
  -/
  use a; rw [← ha, @preimage_comp _ _ _ Subtype.val f]
  /-
    case h
    β α : Type u
    f : β → α
    s : Set β
    hθ : LE.le (Cardinal.mk ↑s) (Cardinal.mk β)
    h₁ : LE.le Cardinal.aleph0 (Cardinal.mk ↑s)
    h₂ : LT.lt (Cardinal.mk α) (Cardinal.mk ↑s).ord.cof
    a : α
    ha : Eq (Cardinal.mk ↑(Set.preimage (Function.comp f Subtype.val) (Singleton.s …
    ⊢ LE.le (Cardinal.mk ↑(Set.preimage Subtype.val (Set.preimage f (Singleton.sin …
  -/
  exact mk_preimage_of_injective _ _ Subtype.val_injective
  /-
    🎉 no goals
  -/


theorem infinite_pigeonhole_set {β α : Type u} {s : Set β} (f : s → α) (θ : Cardinal)
    (hθ : θ ≤ #s) (h₁ : ℵ₀ ≤ θ) (h₂ : #α < θ.ord.cof) :
    ∃ (a : α) (t : Set β) (h : t ⊆ s), θ ≤ #t ∧ ∀ ⦃x⦄ (hx : x ∈ t), f ⟨x, h hx⟩ = a := by
  /-
    β α : Type u
    s : Set β
    f : ↑s → α
    θ : Cardinal.{u}
    hθ : LE.le θ (Cardinal.mk ↑s)
    h₁ : LE.le Cardinal.aleph0 θ
    h₂ : LT.lt (Cardinal.mk α) θ.ord.cof
    ⊢ Exists fun a => Exists fun t => Exists fun h => And (LE.le θ (Cardinal.mk ↑t …
  -/
  cases' infinite_pigeonhole_card f θ hθ h₁ h₂ with a ha
  /-
    case intro
    β α : Type u
    s : Set β
    f : ↑s → α
    θ : Cardinal.{u}
    hθ : LE.le θ (Cardinal.mk ↑s)
    h₁ : LE.le Cardinal.aleph0 θ
    h₂ : LT.lt (Cardinal.mk α) θ.ord.cof
    a : α
    ha : LE.le θ (Cardinal.mk ↑(Set.preimage f (Singleton.singleton a)))
    ⊢ Exists fun a => Exists fun t => Exists fun h => And (LE.le θ (Cardinal.mk ↑t …
  -/
  refine ⟨a, { x | ∃ h, f ⟨x, h⟩ = a }, ?_, ?_, ?_⟩
    /-
      case intro.refine_1
      β α : Type u
      s : Set β
      f : ↑s → α
      θ : Cardinal.{u}
      hθ : LE.le θ (Cardinal.mk ↑s)
      h₁ : LE.le Cardinal.aleph0 θ
      h₂ : LT.lt (Cardinal.mk α) θ.ord.cof
      a : α
      ha : LE.le θ (Cardinal.mk ↑(Set.preimage f (Singleton.singleton a)))
      ⊢ HasSubset.Subset (setOf fun x => Exists fun h => Eq (f ⟨x, h⟩) a) s
    -/
  · rintro x ⟨hx, _⟩
    /-
      case intro.refine_1.intro
      β α : Type u
      s : Set β
      f : ↑s → α
      θ : Cardinal.{u}
      hθ : LE.le θ (Cardinal.mk ↑s)
      h₁ : LE.le Cardinal.aleph0 θ
      h₂ : LT.lt (Cardinal.mk α) θ.ord.cof
      a : α
      ha : LE.le θ (Cardinal.mk ↑(Set.preimage f (Singleton.singleton a)))
      x : β
      hx : Membership.mem s x
      h✝ : Eq (f ⟨x, hx⟩) a
      ⊢ Membership.mem s x
    -/
    exact hx
    /-
      🎉 no goals
    -/
  · refine
      ha.trans
        (ge_of_eq <|
          Quotient.sound ⟨Equiv.trans ?_ (Equiv.subtypeSubtypeEquivSubtypeExists _ _).symm⟩)
    /-
      case intro.refine_2
      β α : Type u
      s : Set β
      f : ↑s → α
      θ : Cardinal.{u}
      hθ : LE.le θ (Cardinal.mk ↑s)
      h₁ : LE.le Cardinal.aleph0 θ
      h₂ : LT.lt (Cardinal.mk α) θ.ord.cof
      a : α
      ha : LE.le θ (Cardinal.mk ↑(Set.preimage f (Singleton.singleton a)))
      ⊢ Equiv (↑(setOf fun x => Exists fun h => Eq (f ⟨x, h⟩) a)) (Subtype fun a_1 = …
    -/
    simp only [coe_eq_subtype, mem_singleton_iff, mem_preimage, mem_setOf_eq]
    /-
      case intro.refine_2
      β α : Type u
      s : Set β
      f : ↑s → α
      θ : Cardinal.{u}
      hθ : LE.le θ (Cardinal.mk ↑s)
      h₁ : LE.le Cardinal.aleph0 θ
      h₂ : LT.lt (Cardinal.mk α) θ.ord.cof
      a : α
      ha : LE.le θ (Cardinal.mk ↑(Set.preimage f (Singleton.singleton a)))
      ⊢ Equiv (Subtype fun x => Exists fun h => Eq (f ⟨x, h⟩) a) (Subtype fun a_1 => …
    -/
    rfl
    /-
      🎉 no goals
    -/
  /-
    case intro.refine_3
    β α : Type u
    s : Set β
    f : ↑s → α
    θ : Cardinal.{u}
    hθ : LE.le θ (Cardinal.mk ↑s)
    h₁ : LE.le Cardinal.aleph0 θ
    h₂ : LT.lt (Cardinal.mk α) θ.ord.cof
    a : α
    ha : LE.le θ (Cardinal.mk ↑(Set.preimage f (Singleton.singleton a)))
    ⊢ ∀ ⦃x : β⦄ (hx : Membership.mem (setOf fun x => Exists fun h => Eq (f ⟨x, h⟩) …
  -/
  rintro x ⟨_, hx'⟩; exact hx'
                     /-
                       🎉 no goals
                     -/


/-- A cardinal is a strong limit if it is not zero and it is
  closed under powersets. Note that `ℵ₀` is a strong limit by this definition. -/
def IsStrongLimit (c : Cardinal) : Prop :=
  c ≠ 0 ∧ ∀ x < c, (2^x) < c


theorem IsStrongLimit.ne_zero {c} (h : IsStrongLimit c) : c ≠ 0 :=
  h.1


theorem IsStrongLimit.two_power_lt {x c} (h : IsStrongLimit c) : x < c → (2^x) < c :=
  h.2 x


theorem isStrongLimit_aleph0 : IsStrongLimit ℵ₀ :=
  ⟨aleph0_ne_zero, fun x hx => by
    /-
      x : Cardinal.{u_1}
      hx : LT.lt x Cardinal.aleph0
      ⊢ LT.lt (HPow.hPow 2 x) Cardinal.aleph0
    -/
    rcases lt_aleph0.1 hx with ⟨n, rfl⟩
    /-
      case intro
      n : Nat
      hx : LT.lt (↑n) Cardinal.aleph0
      ⊢ LT.lt (HPow.hPow 2 ↑n) Cardinal.aleph0
    -/
    exact mod_cast nat_lt_aleph0 (2 ^ n)⟩
    /-
      🎉 no goals
    -/


protected theorem IsStrongLimit.isSuccLimit {c} (H : IsStrongLimit c) : IsSuccLimit c := by
  /-
    c : Cardinal.{u_1}
    H : c.IsStrongLimit
    ⊢ Order.IsSuccLimit c
  -/
  rw [Cardinal.isSuccLimit_iff]
  exact ⟨H.ne_zero, isSuccPrelimit_of_succ_lt fun x h =>
    (succ_le_of_lt <| cantor x).trans_lt (H.two_power_lt h)⟩


protected theorem IsStrongLimit.isSuccPrelimit {c} (H : IsStrongLimit c) : IsSuccPrelimit c :=
  H.isSuccLimit.isSuccPrelimit


theorem IsStrongLimit.aleph0_le {c} (H : IsStrongLimit c) : ℵ₀ ≤ c :=
  aleph0_le_of_isSuccLimit H.isSuccLimit


set_option linter.deprecated false in
@[deprecated IsStrongLimit.isSuccLimit (since := "2024-09-17")]
theorem IsStrongLimit.isLimit {c} (H : IsStrongLimit c) : IsLimit c :=
  ⟨H.ne_zero, H.isSuccPrelimit⟩


theorem isStrongLimit_beth {o : Ordinal} (H : IsSuccPrelimit o) : IsStrongLimit (ℶ_ o) := by
  /-
    o : Ordinal.{u_1}
    H : Order.IsSuccPrelimit o
    ⊢ (Cardinal.beth o).IsStrongLimit
  -/
  rcases eq_or_ne o 0 with (rfl | h)
    /-
      case inl
      H : Order.IsSuccPrelimit 0
      ⊢ (Cardinal.beth 0).IsStrongLimit
    -/
  · rw [beth_zero]
    /-
      case inl
      H : Order.IsSuccPrelimit 0
      ⊢ Cardinal.aleph0.IsStrongLimit
    -/
    exact isStrongLimit_aleph0
    /-
      🎉 no goals
    -/
    /-
      case inr
      o : Ordinal.{u_1}
      H : Order.IsSuccPrelimit o
      h : Ne o 0
      ⊢ (Cardinal.beth o).IsStrongLimit
    -/
  · refine ⟨beth_ne_zero o, fun a ha => ?_⟩
    /-
      case inr
      o : Ordinal.{u_1}
      H : Order.IsSuccPrelimit o
      h : Ne o 0
      a : Cardinal.{u_1}
      ha : LT.lt a (Cardinal.beth o)
      ⊢ LT.lt (HPow.hPow 2 a) (Cardinal.beth o)
    -/
    rw [beth_limit] at ha
      /-
        case inr
        o : Ordinal.{u_1}
        H : Order.IsSuccPrelimit o
        h : Ne o 0
        a : Cardinal.{u_1}
        ha : LT.lt a (iSup fun a => Cardinal.beth ↑a)
        ⊢ LT.lt (HPow.hPow 2 a) (Cardinal.beth o)
      -/
    · rcases exists_lt_of_lt_ciSup' ha with ⟨⟨i, hi⟩, ha⟩
      /-
        case inr.intro.mk
        o : Ordinal.{u_1}
        H : Order.IsSuccPrelimit o
        h : Ne o 0
        a : Cardinal.{u_1}
        ha✝ : LT.lt a (iSup fun a => Cardinal.beth ↑a)
        i : Ordinal.{u_1}
        hi : Membership.mem (Set.Iio o) i
        ha : LT.lt a (Cardinal.beth ↑⟨i, hi⟩)
        ⊢ LT.lt (HPow.hPow 2 a) (Cardinal.beth o)
      -/
      have := power_le_power_left two_ne_zero ha.le
      /-
        case inr.intro.mk
        o : Ordinal.{u_1}
        H : Order.IsSuccPrelimit o
        h : Ne o 0
        a : Cardinal.{u_1}
        ha✝ : LT.lt a (iSup fun a => Cardinal.beth ↑a)
        i : Ordinal.{u_1}
        hi : Membership.mem (Set.Iio o) i
        ha : LT.lt a (Cardinal.beth ↑⟨i, hi⟩)
        this : LE.le (HPow.hPow 2 a) (HPow.hPow 2 (Cardinal.beth ↑⟨i, hi⟩))
        ⊢ LT.lt (HPow.hPow 2 a) (Cardinal.beth o)
      -/
      rw [← beth_succ] at this
      /-
        case inr.intro.mk
        o : Ordinal.{u_1}
        H : Order.IsSuccPrelimit o
        h : Ne o 0
        a : Cardinal.{u_1}
        ha✝ : LT.lt a (iSup fun a => Cardinal.beth ↑a)
        i : Ordinal.{u_1}
        hi : Membership.mem (Set.Iio o) i
        ha : LT.lt a (Cardinal.beth ↑⟨i, hi⟩)
        this : LE.le (HPow.hPow 2 a) (Cardinal.beth (Order.succ ↑⟨i, hi⟩))
        ⊢ LT.lt (HPow.hPow 2 a) (Cardinal.beth o)
      -/
      exact this.trans_lt (beth_lt.2 (H.succ_lt hi))
      /-
        🎉 no goals
      -/
      /-
        case inr
        o : Ordinal.{u_1}
        H : Order.IsSuccPrelimit o
        h : Ne o 0
        a : Cardinal.{u_1}
        ha : LT.lt a (Cardinal.beth o)
        ⊢ o.IsLimit
      -/
    · rw [isLimit_iff]
      /-
        case inr
        o : Ordinal.{u_1}
        H : Order.IsSuccPrelimit o
        h : Ne o 0
        a : Cardinal.{u_1}
        ha : LT.lt a (Cardinal.beth o)
        ⊢ And (Ne o 0) (Order.IsSuccPrelimit o)
      -/
      exact ⟨h, H⟩
      /-
        🎉 no goals
      -/


theorem mk_bounded_subset {α : Type*} (h : ∀ x < #α, (2^x) < #α) {r : α → α → Prop}
    [IsWellOrder α r] (hr : (#α).ord = type r) : #{ s : Set α // Bounded r s } = #α := by
  /-
    α : Type u_1
    h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
    ⊢ Eq (Cardinal.mk (Subtype fun s => Set.Bounded r s)) (Cardinal.mk α)
  -/
  rcases eq_or_ne #α 0 with (ha | ha)
    /-
      case inl
      α : Type u_1
      h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
      ha : Eq (Cardinal.mk α) 0
      ⊢ Eq (Cardinal.mk (Subtype fun s => Set.Bounded r s)) (Cardinal.mk α)
    -/
  · rw [ha]
    /-
      case inl
      α : Type u_1
      h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
      ha : Eq (Cardinal.mk α) 0
      ⊢ Eq (Cardinal.mk (Subtype fun s => Set.Bounded r s)) 0
    -/
    haveI := mk_eq_zero_iff.1 ha
    /-
      case inl
      α : Type u_1
      h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
      ha : Eq (Cardinal.mk α) 0
      this : IsEmpty α
      ⊢ Eq (Cardinal.mk (Subtype fun s => Set.Bounded r s)) 0
    -/
    rw [mk_eq_zero_iff]
    /-
      case inl
      α : Type u_1
      h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
      ha : Eq (Cardinal.mk α) 0
      this : IsEmpty α
      ⊢ IsEmpty (Subtype fun s => Set.Bounded r s)
    -/
    constructor
    /-
      case inl.false
      α : Type u_1
      h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
      ha : Eq (Cardinal.mk α) 0
      this : IsEmpty α
      ⊢ (Subtype fun s => Set.Bounded r s) → False
    -/
    rintro ⟨s, hs⟩
    /-
      case inl.false.mk
      α : Type u_1
      h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
      ha : Eq (Cardinal.mk α) 0
      this : IsEmpty α
      s : Set α
      hs : Set.Bounded r s
      ⊢ False
    -/
    exact (not_unbounded_iff s).2 hs (unbounded_of_isEmpty s)
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
    ha : Ne (Cardinal.mk α) 0
    ⊢ Eq (Cardinal.mk (Subtype fun s => Set.Bounded r s)) (Cardinal.mk α)
  -/
  have h' : IsStrongLimit #α := ⟨ha, h⟩
  /-
    case inr
    α : Type u_1
    h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
    ha : Ne (Cardinal.mk α) 0
    h' : (Cardinal.mk α).IsStrongLimit
    ⊢ Eq (Cardinal.mk (Subtype fun s => Set.Bounded r s)) (Cardinal.mk α)
  -/
  have ha := h'.aleph0_le
  /-
    case inr
    α : Type u_1
    h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
    ha✝ : Ne (Cardinal.mk α) 0
    h' : (Cardinal.mk α).IsStrongLimit
    ha : LE.le Cardinal.aleph0 (Cardinal.mk α)
    ⊢ Eq (Cardinal.mk (Subtype fun s => Set.Bounded r s)) (Cardinal.mk α)
  -/
  apply le_antisymm
    /-
      case inr.a
      α : Type u_1
      h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
      ha✝ : Ne (Cardinal.mk α) 0
      h' : (Cardinal.mk α).IsStrongLimit
      ha : LE.le Cardinal.aleph0 (Cardinal.mk α)
      ⊢ LE.le (Cardinal.mk (Subtype fun s => Set.Bounded r s)) (Cardinal.mk α)
    -/
  · have : { s : Set α | Bounded r s } = ⋃ i, 𝒫{ j | r j i } := setOf_exists _
    /-
      case inr.a
      α : Type u_1
      h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
      ha✝ : Ne (Cardinal.mk α) 0
      h' : (Cardinal.mk α).IsStrongLimit
      ha : LE.le Cardinal.aleph0 (Cardinal.mk α)
      this : Eq (setOf fun s => Set.Bounded r s) (Set.iUnion fun i => (setOf fun j = …
      ⊢ LE.le (Cardinal.mk (Subtype fun s => Set.Bounded r s)) (Cardinal.mk α)
    -/
    rw [← coe_setOf, this]
    refine mk_iUnion_le_sum_mk.trans ((sum_le_iSup (fun i => #(𝒫{ j | r j i }))).trans
      ((mul_le_max_of_aleph0_le_left ha).trans ?_))
    /-
      case inr.a
      α : Type u_1
      h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
      ha✝ : Ne (Cardinal.mk α) 0
      h' : (Cardinal.mk α).IsStrongLimit
      ha : LE.le Cardinal.aleph0 (Cardinal.mk α)
      this : Eq (setOf fun s => Set.Bounded r s) (Set.iUnion fun i => (setOf fun j = …
      ⊢ LE.le (Max.max (Cardinal.mk α) (iSup fun i => Cardinal.mk ↑(setOf fun j => r …
    -/
    rw [max_eq_left]
    /-
      case inr.a
      α : Type u_1
      h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
      ha✝ : Ne (Cardinal.mk α) 0
      h' : (Cardinal.mk α).IsStrongLimit
      ha : LE.le Cardinal.aleph0 (Cardinal.mk α)
      this : Eq (setOf fun s => Set.Bounded r s) (Set.iUnion fun i => (setOf fun j = …
      ⊢ LE.le (iSup fun i => Cardinal.mk ↑(setOf fun j => r j i).powerset) (Cardinal …
    -/
    apply ciSup_le' _
    /-
      α : Type u_1
      h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
      ha✝ : Ne (Cardinal.mk α) 0
      h' : (Cardinal.mk α).IsStrongLimit
      ha : LE.le Cardinal.aleph0 (Cardinal.mk α)
      this : Eq (setOf fun s => Set.Bounded r s) (Set.iUnion fun i => (setOf fun j = …
      ⊢ ∀ (i : α), LE.le (Cardinal.mk ↑(setOf fun j => r j i).powerset) (Cardinal.mk …
    -/
    intro i
    /-
      α : Type u_1
      h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
      ha✝ : Ne (Cardinal.mk α) 0
      h' : (Cardinal.mk α).IsStrongLimit
      ha : LE.le Cardinal.aleph0 (Cardinal.mk α)
      this : Eq (setOf fun s => Set.Bounded r s) (Set.iUnion fun i => (setOf fun j = …
      i : α
      ⊢ LE.le (Cardinal.mk ↑(setOf fun j => r j i).powerset) (Cardinal.mk α)
    -/
    rw [mk_powerset]
    /-
      α : Type u_1
      h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
      ha✝ : Ne (Cardinal.mk α) 0
      h' : (Cardinal.mk α).IsStrongLimit
      ha : LE.le Cardinal.aleph0 (Cardinal.mk α)
      this : Eq (setOf fun s => Set.Bounded r s) (Set.iUnion fun i => (setOf fun j = …
      i : α
      ⊢ LE.le (HPow.hPow 2 (Cardinal.mk ↑(setOf fun j => r j i))) (Cardinal.mk α)
    -/
    apply (h'.two_power_lt _).le
    /-
      α : Type u_1
      h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
      ha✝ : Ne (Cardinal.mk α) 0
      h' : (Cardinal.mk α).IsStrongLimit
      ha : LE.le Cardinal.aleph0 (Cardinal.mk α)
      this : Eq (setOf fun s => Set.Bounded r s) (Set.iUnion fun i => (setOf fun j = …
      i : α
      ⊢ LT.lt (Cardinal.mk ↑(setOf fun j => r j i)) (Cardinal.mk α)
    -/
    rw [coe_setOf, card_typein, ← lt_ord, hr]
    /-
      α : Type u_1
      h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
      ha✝ : Ne (Cardinal.mk α) 0
      h' : (Cardinal.mk α).IsStrongLimit
      ha : LE.le Cardinal.aleph0 (Cardinal.mk α)
      this : Eq (setOf fun s => Set.Bounded r s) (Set.iUnion fun i => (setOf fun j = …
      i : α
      ⊢ LT.lt ((Ordinal.typein r).toRelEmbedding i) (Ordinal.type r)
    -/
    apply typein_lt_type
    /-
      🎉 no goals
    -/
    /-
      case inr.a
      α : Type u_1
      h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
      ha✝ : Ne (Cardinal.mk α) 0
      h' : (Cardinal.mk α).IsStrongLimit
      ha : LE.le Cardinal.aleph0 (Cardinal.mk α)
      ⊢ LE.le (Cardinal.mk α) (Cardinal.mk (Subtype fun s => Set.Bounded r s))
    -/
  · refine @mk_le_of_injective α _ (fun x => Subtype.mk {x} ?_) ?_
      /-
        case inr.a.refine_1
        α : Type u_1
        h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
        r : α → α → Prop
        inst✝ : IsWellOrder α r
        hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
        ha✝ : Ne (Cardinal.mk α) 0
        h' : (Cardinal.mk α).IsStrongLimit
        ha : LE.le Cardinal.aleph0 (Cardinal.mk α)
        x : α
        ⊢ Set.Bounded r (Singleton.singleton x)
      -/
    · apply bounded_singleton
      /-
        case inr.a.refine_1.hr
        α : Type u_1
        h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
        r : α → α → Prop
        inst✝ : IsWellOrder α r
        hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
        ha✝ : Ne (Cardinal.mk α) 0
        h' : (Cardinal.mk α).IsStrongLimit
        ha : LE.le Cardinal.aleph0 (Cardinal.mk α)
        x : α
        ⊢ (Ordinal.type r).IsLimit
      -/
      rw [← hr]
      /-
        case inr.a.refine_1.hr
        α : Type u_1
        h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
        r : α → α → Prop
        inst✝ : IsWellOrder α r
        hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
        ha✝ : Ne (Cardinal.mk α) 0
        h' : (Cardinal.mk α).IsStrongLimit
        ha : LE.le Cardinal.aleph0 (Cardinal.mk α)
        x : α
        ⊢ (Cardinal.mk α).ord.IsLimit
      -/
      apply isLimit_ord ha
      /-
        🎉 no goals
      -/
      /-
        case inr.a.refine_2
        α : Type u_1
        h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
        r : α → α → Prop
        inst✝ : IsWellOrder α r
        hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
        ha✝ : Ne (Cardinal.mk α) 0
        h' : (Cardinal.mk α).IsStrongLimit
        ha : LE.le Cardinal.aleph0 (Cardinal.mk α)
        ⊢ Function.Injective fun x => ⟨Singleton.singleton x, ⋯⟩
      -/
    · intro a b hab
      /-
        case inr.a.refine_2
        α : Type u_1
        h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
        r : α → α → Prop
        inst✝ : IsWellOrder α r
        hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
        ha✝ : Ne (Cardinal.mk α) 0
        h' : (Cardinal.mk α).IsStrongLimit
        ha : LE.le Cardinal.aleph0 (Cardinal.mk α)
        a b : α
        hab : Eq ((fun x => ⟨Singleton.singleton x, ⋯⟩) a) ((fun x => ⟨Singleton.singl …
        ⊢ Eq a b
      -/
      simpa [singleton_eq_singleton_iff] using hab
      /-
        🎉 no goals
      -/


theorem mk_subset_mk_lt_cof {α : Type*} (h : ∀ x < #α, (2^x) < #α) :
    #{ s : Set α // #s < cof (#α).ord } = #α := by
  /-
    α : Type u_1
    h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
    ⊢ Eq (Cardinal.mk (Subtype fun s => LT.lt (Cardinal.mk ↑s) (Cardinal.mk α).ord …
  -/
  rcases eq_or_ne #α 0 with (ha | ha)
    /-
      case inl
      α : Type u_1
      h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
      ha : Eq (Cardinal.mk α) 0
      ⊢ Eq (Cardinal.mk (Subtype fun s => LT.lt (Cardinal.mk ↑s) (Cardinal.mk α).ord …
    -/
  · simp [ha]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
    ha : Ne (Cardinal.mk α) 0
    ⊢ Eq (Cardinal.mk (Subtype fun s => LT.lt (Cardinal.mk ↑s) (Cardinal.mk α).ord …
  -/
  have h' : IsStrongLimit #α := ⟨ha, h⟩
  /-
    case inr
    α : Type u_1
    h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
    ha : Ne (Cardinal.mk α) 0
    h' : (Cardinal.mk α).IsStrongLimit
    ⊢ Eq (Cardinal.mk (Subtype fun s => LT.lt (Cardinal.mk ↑s) (Cardinal.mk α).ord …
  -/
  rcases ord_eq α with ⟨r, wo, hr⟩
  /-
    case inr.intro.intro
    α : Type u_1
    h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
    ha : Ne (Cardinal.mk α) 0
    h' : (Cardinal.mk α).IsStrongLimit
    r : α → α → Prop
    wo : IsWellOrder α r
    hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
    ⊢ Eq (Cardinal.mk (Subtype fun s => LT.lt (Cardinal.mk ↑s) (Cardinal.mk α).ord …
  -/
  haveI := wo
  /-
    case inr.intro.intro
    α : Type u_1
    h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
    ha : Ne (Cardinal.mk α) 0
    h' : (Cardinal.mk α).IsStrongLimit
    r : α → α → Prop
    wo : IsWellOrder α r
    hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
    this : IsWellOrder α r
    ⊢ Eq (Cardinal.mk (Subtype fun s => LT.lt (Cardinal.mk ↑s) (Cardinal.mk α).ord …
  -/
  apply le_antisymm
    /-
      case inr.intro.intro.a
      α : Type u_1
      h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
      ha : Ne (Cardinal.mk α) 0
      h' : (Cardinal.mk α).IsStrongLimit
      r : α → α → Prop
      wo : IsWellOrder α r
      hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
      this : IsWellOrder α r
      ⊢ LE.le (Cardinal.mk (Subtype fun s => LT.lt (Cardinal.mk ↑s) (Cardinal.mk α). …
    -/
  · conv_rhs => rw [← mk_bounded_subset h hr]
    /-
      case inr.intro.intro.a
      α : Type u_1
      h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
      ha : Ne (Cardinal.mk α) 0
      h' : (Cardinal.mk α).IsStrongLimit
      r : α → α → Prop
      wo : IsWellOrder α r
      hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
      this : IsWellOrder α r
      ⊢ LE.le (Cardinal.mk (Subtype fun s => LT.lt (Cardinal.mk ↑s) (Cardinal.mk α). …
    -/
    apply mk_le_mk_of_subset
    /-
      case inr.intro.intro.a.h
      α : Type u_1
      h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
      ha : Ne (Cardinal.mk α) 0
      h' : (Cardinal.mk α).IsStrongLimit
      r : α → α → Prop
      wo : IsWellOrder α r
      hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
      this : IsWellOrder α r
      ⊢ HasSubset.Subset (fun x => And (Quotient.liftOn₂ (Cardinal.mk ↑x) (Cardinal. …
    -/
    intro s hs
    /-
      case inr.intro.intro.a.h
      α : Type u_1
      h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
      ha : Ne (Cardinal.mk α) 0
      h' : (Cardinal.mk α).IsStrongLimit
      r : α → α → Prop
      wo : IsWellOrder α r
      hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
      this : IsWellOrder α r
      s : Set α
      hs : Membership.mem (fun x => And (Quotient.liftOn₂ (Cardinal.mk ↑x) (Cardinal …
      ⊢ Membership.mem (fun x => Exists fun a => ∀ (b : α), Membership.mem x b → r b …
    -/
    rw [hr] at hs
    /-
      case inr.intro.intro.a.h
      α : Type u_1
      h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
      ha : Ne (Cardinal.mk α) 0
      h' : (Cardinal.mk α).IsStrongLimit
      r : α → α → Prop
      wo : IsWellOrder α r
      hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
      this : IsWellOrder α r
      s : Set α
      hs : Membership.mem (fun x => And (Quotient.liftOn₂ (Cardinal.mk ↑x) (Ordinal. …
      ⊢ Membership.mem (fun x => Exists fun a => ∀ (b : α), Membership.mem x b → r b …
    -/
    exact lt_cof_type hs
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.intro.a
      α : Type u_1
      h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
      ha : Ne (Cardinal.mk α) 0
      h' : (Cardinal.mk α).IsStrongLimit
      r : α → α → Prop
      wo : IsWellOrder α r
      hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
      this : IsWellOrder α r
      ⊢ LE.le (Cardinal.mk α) (Cardinal.mk (Subtype fun s => LT.lt (Cardinal.mk ↑s)  …
    -/
  · refine @mk_le_of_injective α _ (fun x => Subtype.mk {x} ?_) ?_
      /-
        case inr.intro.intro.a.refine_1
        α : Type u_1
        h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
        ha : Ne (Cardinal.mk α) 0
        h' : (Cardinal.mk α).IsStrongLimit
        r : α → α → Prop
        wo : IsWellOrder α r
        hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
        this : IsWellOrder α r
        x : α
        ⊢ LT.lt (Cardinal.mk ↑(Singleton.singleton x)) (Cardinal.mk α).ord.cof
      -/
    · rw [mk_singleton]
      /-
        case inr.intro.intro.a.refine_1
        α : Type u_1
        h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
        ha : Ne (Cardinal.mk α) 0
        h' : (Cardinal.mk α).IsStrongLimit
        r : α → α → Prop
        wo : IsWellOrder α r
        hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
        this : IsWellOrder α r
        x : α
        ⊢ LT.lt 1 (Cardinal.mk α).ord.cof
      -/
      exact one_lt_aleph0.trans_le (aleph0_le_cof.2 (isLimit_ord h'.aleph0_le))
      /-
        🎉 no goals
      -/
      /-
        case inr.intro.intro.a.refine_2
        α : Type u_1
        h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
        ha : Ne (Cardinal.mk α) 0
        h' : (Cardinal.mk α).IsStrongLimit
        r : α → α → Prop
        wo : IsWellOrder α r
        hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
        this : IsWellOrder α r
        ⊢ Function.Injective fun x => ⟨Singleton.singleton x, ⋯⟩
      -/
    · intro a b hab
      /-
        case inr.intro.intro.a.refine_2
        α : Type u_1
        h : ∀ (x : Cardinal.{u_1}), LT.lt x (Cardinal.mk α) → LT.lt (HPow.hPow 2 x) (C …
        ha : Ne (Cardinal.mk α) 0
        h' : (Cardinal.mk α).IsStrongLimit
        r : α → α → Prop
        wo : IsWellOrder α r
        hr : Eq (Cardinal.mk α).ord (Ordinal.type r)
        this : IsWellOrder α r
        a b : α
        hab : Eq ((fun x => ⟨Singleton.singleton x, ⋯⟩) a) ((fun x => ⟨Singleton.singl …
        ⊢ Eq a b
      -/
      simpa [singleton_eq_singleton_iff] using hab
      /-
        🎉 no goals
      -/


/-- A cardinal is regular if it is infinite and it equals its own cofinality. -/
def IsRegular (c : Cardinal) : Prop :=
  ℵ₀ ≤ c ∧ c ≤ c.ord.cof


theorem IsRegular.aleph0_le {c : Cardinal} (H : c.IsRegular) : ℵ₀ ≤ c :=
  H.1


theorem IsRegular.cof_eq {c : Cardinal} (H : c.IsRegular) : c.ord.cof = c :=
  (cof_ord_le c).antisymm H.2


theorem IsRegular.cof_omega_eq {o : Ordinal} (H : (ℵ_ o).IsRegular) : (ω_ o).cof = ℵ_ o := by
  /-
    o : Ordinal.{u_1}
    H : (Cardinal.aleph o).IsRegular
    ⊢ Eq (Ordinal.omega o).cof (Cardinal.aleph o)
  -/
  rw [← ord_aleph, H.cof_eq]
  /-
    🎉 no goals
  -/


theorem IsRegular.pos {c : Cardinal} (H : c.IsRegular) : 0 < c :=
  aleph0_pos.trans_le H.1


theorem IsRegular.nat_lt {c : Cardinal} (H : c.IsRegular) (n : ℕ) : n < c :=
  lt_of_lt_of_le (nat_lt_aleph0 n) H.aleph0_le


theorem IsRegular.ord_pos {c : Cardinal} (H : c.IsRegular) : 0 < c.ord := by
  /-
    c : Cardinal.{u_1}
    H : c.IsRegular
    ⊢ LT.lt 0 c.ord
  -/
  rw [Cardinal.lt_ord, card_zero]
  /-
    c : Cardinal.{u_1}
    H : c.IsRegular
    ⊢ LT.lt 0 c
  -/
  exact H.pos
  /-
    🎉 no goals
  -/


theorem isRegular_cof {o : Ordinal} (h : o.IsLimit) : IsRegular o.cof :=
  ⟨aleph0_le_cof.2 h, (cof_cof o).ge⟩


theorem isRegular_aleph0 : IsRegular ℵ₀ :=
              /-
                ⊢ LE.le Cardinal.aleph0 Cardinal.aleph0.ord.cof
              -/
  ⟨le_rfl, by simp⟩
              /-
                🎉 no goals
              -/


theorem isRegular_succ {c : Cardinal.{u}} (h : ℵ₀ ≤ c) : IsRegular (succ c) :=
  ⟨h.trans (le_succ c),
    succ_le_of_lt
      (by
        /-
          c : Cardinal.{u}
          h : LE.le Cardinal.aleph0 c
          ⊢ LT.lt c (Order.succ c).ord.cof
        -/
        have αe := Cardinal.mk_out (succ c)
        /-
          c : Cardinal.{u}
          h : LE.le Cardinal.aleph0 c
          αe : Eq (Cardinal.mk (Quotient.out (Order.succ c))) (Order.succ c)
          ⊢ LT.lt c (Order.succ c).ord.cof
        -/
        set α := (succ c).out
        /-
          c : Cardinal.{u}
          h : LE.le Cardinal.aleph0 c
          α : Type u := Quotient.out (Order.succ c)
          αe : Eq (Cardinal.mk α) (Order.succ c)
          ⊢ LT.lt c (Order.succ c).ord.cof
        -/
        rcases ord_eq α with ⟨r, wo, re⟩
        /-
          case intro.intro
          c : Cardinal.{u}
          h : LE.le Cardinal.aleph0 c
          α : Type u := Quotient.out (Order.succ c)
          αe : Eq (Cardinal.mk α) (Order.succ c)
          r : α → α → Prop
          wo : IsWellOrder α r
          re : Eq (Cardinal.mk α).ord (Ordinal.type r)
          ⊢ LT.lt c (Order.succ c).ord.cof
        -/
        have := isLimit_ord (h.trans (le_succ _))
        /-
          case intro.intro
          c : Cardinal.{u}
          h : LE.le Cardinal.aleph0 c
          α : Type u := Quotient.out (Order.succ c)
          αe : Eq (Cardinal.mk α) (Order.succ c)
          r : α → α → Prop
          wo : IsWellOrder α r
          re : Eq (Cardinal.mk α).ord (Ordinal.type r)
          this : (Order.succ c).ord.IsLimit
          ⊢ LT.lt c (Order.succ c).ord.cof
        -/
        rw [← αe, re] at this ⊢
        /-
          case intro.intro
          c : Cardinal.{u}
          h : LE.le Cardinal.aleph0 c
          α : Type u := Quotient.out (Order.succ c)
          αe : Eq (Cardinal.mk α) (Order.succ c)
          r : α → α → Prop
          wo : IsWellOrder α r
          re : Eq (Cardinal.mk α).ord (Ordinal.type r)
          this : (Ordinal.type r).IsLimit
          ⊢ LT.lt c (Ordinal.type r).cof
        -/
        rcases cof_eq' r this with ⟨S, H, Se⟩
        /-
          case intro.intro.intro.intro
          c : Cardinal.{u}
          h : LE.le Cardinal.aleph0 c
          α : Type u := Quotient.out (Order.succ c)
          αe : Eq (Cardinal.mk α) (Order.succ c)
          r : α → α → Prop
          wo : IsWellOrder α r
          re : Eq (Cardinal.mk α).ord (Ordinal.type r)
          this : (Ordinal.type r).IsLimit
          S : Set α
          H : ∀ (a : α), Exists fun b => And (Membership.mem S b) (r a b)
          Se : Eq (Cardinal.mk ↑S) (Ordinal.type r).cof
          ⊢ LT.lt c (Ordinal.type r).cof
        -/
        rw [← Se]
        /-
          case intro.intro.intro.intro
          c : Cardinal.{u}
          h : LE.le Cardinal.aleph0 c
          α : Type u := Quotient.out (Order.succ c)
          αe : Eq (Cardinal.mk α) (Order.succ c)
          r : α → α → Prop
          wo : IsWellOrder α r
          re : Eq (Cardinal.mk α).ord (Ordinal.type r)
          this : (Ordinal.type r).IsLimit
          S : Set α
          H : ∀ (a : α), Exists fun b => And (Membership.mem S b) (r a b)
          Se : Eq (Cardinal.mk ↑S) (Ordinal.type r).cof
          ⊢ LT.lt c (Cardinal.mk ↑S)
        -/
        apply lt_imp_lt_of_le_imp_le fun h => mul_le_mul_right' h c
        /-
          case intro.intro.intro.intro
          c : Cardinal.{u}
          h : LE.le Cardinal.aleph0 c
          α : Type u := Quotient.out (Order.succ c)
          αe : Eq (Cardinal.mk α) (Order.succ c)
          r : α → α → Prop
          wo : IsWellOrder α r
          re : Eq (Cardinal.mk α).ord (Ordinal.type r)
          this : (Ordinal.type r).IsLimit
          S : Set α
          H : ∀ (a : α), Exists fun b => And (Membership.mem S b) (r a b)
          Se : Eq (Cardinal.mk ↑S) (Ordinal.type r).cof
          ⊢ LT.lt (HMul.hMul c c) (HMul.hMul (Cardinal.mk ↑S) c)
        -/
        rw [mul_eq_self h, ← succ_le_iff, ← αe, ← sum_const']
        /-
          case intro.intro.intro.intro
          c : Cardinal.{u}
          h : LE.le Cardinal.aleph0 c
          α : Type u := Quotient.out (Order.succ c)
          αe : Eq (Cardinal.mk α) (Order.succ c)
          r : α → α → Prop
          wo : IsWellOrder α r
          re : Eq (Cardinal.mk α).ord (Ordinal.type r)
          this : (Ordinal.type r).IsLimit
          S : Set α
          H : ∀ (a : α), Exists fun b => And (Membership.mem S b) (r a b)
          Se : Eq (Cardinal.mk ↑S) (Ordinal.type r).cof
          ⊢ LE.le (Cardinal.mk α) (Cardinal.sum fun x => c)
        -/
        refine le_trans ?_ (sum_le_sum (fun (x : S) => card (typein r (x : α))) _ fun i => ?_)
          /-
            case intro.intro.intro.intro.refine_1
            c : Cardinal.{u}
            h : LE.le Cardinal.aleph0 c
            α : Type u := Quotient.out (Order.succ c)
            αe : Eq (Cardinal.mk α) (Order.succ c)
            r : α → α → Prop
            wo : IsWellOrder α r
            re : Eq (Cardinal.mk α).ord (Ordinal.type r)
            this : (Ordinal.type r).IsLimit
            S : Set α
            H : ∀ (a : α), Exists fun b => And (Membership.mem S b) (r a b)
            Se : Eq (Cardinal.mk ↑S) (Ordinal.type r).cof
            ⊢ LE.le (Cardinal.mk α) (Cardinal.sum fun x => ((Ordinal.typein r).toRelEmbedd …
          -/
        · simp only [← card_typein, ← mk_sigma]
          exact
            ⟨Embedding.ofSurjective (fun x => x.2.1) fun a =>
                let ⟨b, h, ab⟩ := H a
                ⟨⟨⟨_, h⟩, _, ab⟩, rfl⟩⟩
          /-
            case intro.intro.intro.intro.refine_2
            c : Cardinal.{u}
            h : LE.le Cardinal.aleph0 c
            α : Type u := Quotient.out (Order.succ c)
            αe : Eq (Cardinal.mk α) (Order.succ c)
            r : α → α → Prop
            wo : IsWellOrder α r
            re : Eq (Cardinal.mk α).ord (Ordinal.type r)
            this : (Ordinal.type r).IsLimit
            S : Set α
            H : ∀ (a : α), Exists fun b => And (Membership.mem S b) (r a b)
            Se : Eq (Cardinal.mk ↑S) (Ordinal.type r).cof
            i : ↑S
            ⊢ LE.le ((fun x => ((Ordinal.typein r).toRelEmbedding ↑x).card) i) c
          -/
        · rw [← lt_succ_iff, ← lt_ord, ← αe, re]
          /-
            case intro.intro.intro.intro.refine_2
            c : Cardinal.{u}
            h : LE.le Cardinal.aleph0 c
            α : Type u := Quotient.out (Order.succ c)
            αe : Eq (Cardinal.mk α) (Order.succ c)
            r : α → α → Prop
            wo : IsWellOrder α r
            re : Eq (Cardinal.mk α).ord (Ordinal.type r)
            this : (Ordinal.type r).IsLimit
            S : Set α
            H : ∀ (a : α), Exists fun b => And (Membership.mem S b) (r a b)
            Se : Eq (Cardinal.mk ↑S) (Ordinal.type r).cof
            i : ↑S
            ⊢ LT.lt ((Ordinal.typein r).toRelEmbedding ↑i) (Ordinal.type r)
          -/
          apply typein_lt_type)⟩
          /-
            🎉 no goals
          -/


theorem isRegular_aleph_one : IsRegular ℵ₁ := by
  /-
    ⊢ (Cardinal.aleph 1).IsRegular
  -/
  rw [← succ_aleph0]
  /-
    ⊢ (Order.succ Cardinal.aleph0).IsRegular
  -/
  exact isRegular_succ le_rfl
  /-
    🎉 no goals
  -/


theorem isRegular_preAleph_succ {o : Ordinal} (h : ω ≤ o) : IsRegular (preAleph (succ o)) := by
  /-
    o : Ordinal.{u_1}
    h : LE.le Ordinal.omega0 o
    ⊢ (Cardinal.preAleph (Order.succ o)).IsRegular
  -/
  rw [preAleph_succ]
  /-
    o : Ordinal.{u_1}
    h : LE.le Ordinal.omega0 o
    ⊢ (Order.succ (Cardinal.preAleph o)).IsRegular
  -/
  exact isRegular_succ (aleph0_le_preAleph.2 h)
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated isRegular_preAleph_succ (since := "2024-10-22")]
theorem isRegular_aleph'_succ {o : Ordinal} (h : ω ≤ o) : IsRegular (aleph' (succ o)) := by
  /-
    o : Ordinal.{u_1}
    h : LE.le Ordinal.omega0 o
    ⊢ (Cardinal.aleph' (Order.succ o)).IsRegular
  -/
  rw [aleph'_succ]
  /-
    o : Ordinal.{u_1}
    h : LE.le Ordinal.omega0 o
    ⊢ (Order.succ (Cardinal.aleph' o)).IsRegular
  -/
  exact isRegular_succ (aleph0_le_aleph'.2 h)
  /-
    🎉 no goals
  -/


theorem isRegular_aleph_succ (o : Ordinal) : IsRegular (ℵ_ (succ o)) := by
  /-
    o : Ordinal.{u_1}
    ⊢ (Cardinal.aleph (Order.succ o)).IsRegular
  -/
  rw [aleph_succ]
  /-
    o : Ordinal.{u_1}
    ⊢ (Order.succ (Cardinal.aleph o)).IsRegular
  -/
  exact isRegular_succ (aleph0_le_aleph o)
  /-
    🎉 no goals
  -/


/-- A function whose codomain's cardinality is infinite but strictly smaller than its domain's
has a fiber with cardinality strictly great than the codomain.
-/
theorem infinite_pigeonhole_card_lt {β α : Type u} (f : β → α) (w : #α < #β) (w' : ℵ₀ ≤ #α) :
    ∃ a : α, #α < #(f ⁻¹' {a}) := by
  /-
    β α : Type u
    f : β → α
    w : LT.lt (Cardinal.mk α) (Cardinal.mk β)
    w' : LE.le Cardinal.aleph0 (Cardinal.mk α)
    ⊢ Exists fun a => LT.lt (Cardinal.mk α) (Cardinal.mk ↑(Set.preimage f (Singlet …
  -/
  simp_rw [← succ_le_iff]
  exact
    Ordinal.infinite_pigeonhole_card f (succ #α) (succ_le_of_lt w) (w'.trans (lt_succ _).le)
      ((lt_succ _).trans_le (isRegular_succ w').2.ge)


/-- A function whose codomain's cardinality is infinite but strictly smaller than its domain's
has an infinite fiber.
-/
theorem exists_infinite_fiber {β α : Type u} (f : β → α) (w : #α < #β) (w' : Infinite α) :
    ∃ a : α, Infinite (f ⁻¹' {a}) := by
  /-
    β α : Type u
    f : β → α
    w : LT.lt (Cardinal.mk α) (Cardinal.mk β)
    w' : Infinite α
    ⊢ Exists fun a => Infinite ↑(Set.preimage f (Singleton.singleton a))
  -/
  simp_rw [Cardinal.infinite_iff] at w' ⊢
  /-
    β α : Type u
    f : β → α
    w : LT.lt (Cardinal.mk α) (Cardinal.mk β)
    w' : LE.le Cardinal.aleph0 (Cardinal.mk α)
    ⊢ Exists fun a => LE.le Cardinal.aleph0 (Cardinal.mk ↑(Set.preimage f (Singlet …
  -/
  cases' infinite_pigeonhole_card_lt f w w' with a ha
  /-
    case intro
    β α : Type u
    f : β → α
    w : LT.lt (Cardinal.mk α) (Cardinal.mk β)
    w' : LE.le Cardinal.aleph0 (Cardinal.mk α)
    a : α
    ha : LT.lt (Cardinal.mk α) (Cardinal.mk ↑(Set.preimage f (Singleton.singleton  …
    ⊢ Exists fun a => LE.le Cardinal.aleph0 (Cardinal.mk ↑(Set.preimage f (Singlet …
  -/
  exact ⟨a, w'.trans ha.le⟩
  /-
    🎉 no goals
  -/


/-- If an infinite type `β` can be expressed as a union of finite sets,
then the cardinality of the collection of those finite sets
must be at least the cardinality of `β`.
-/
theorem le_range_of_union_finset_eq_top {α β : Type*} [Infinite β] (f : α → Finset β)
    (w : ⋃ a, (f a : Set β) = ⊤) : #β ≤ #(range f) := by
  have k : _root_.Infinite (range f) := by
    rw [infinite_coe_iff]
    apply mt (union_finset_finite_of_range_finite f)
    rw [w]
    exact infinite_univ
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Infinite β
    f : α → Finset β
    w : Eq (Set.iUnion fun a => ↑(f a)) Top.top
    k : Infinite ↑(Set.range f)
    ⊢ LE.le (Cardinal.mk β) (Cardinal.mk ↑(Set.range f))
  -/
  by_contra h
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Infinite β
    f : α → Finset β
    w : Eq (Set.iUnion fun a => ↑(f a)) Top.top
    k : Infinite ↑(Set.range f)
    h : Not (LE.le (Cardinal.mk β) (Cardinal.mk ↑(Set.range f)))
    ⊢ False
  -/
  simp only [not_le] at h
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Infinite β
    f : α → Finset β
    w : Eq (Set.iUnion fun a => ↑(f a)) Top.top
    k : Infinite ↑(Set.range f)
    h : LT.lt (Cardinal.mk ↑(Set.range f)) (Cardinal.mk β)
    ⊢ False
  -/
  let u : ∀ b, ∃ a, b ∈ f a := fun b => by simpa using (w.ge : _) (Set.mem_univ b)
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Infinite β
    f : α → Finset β
    w : Eq (Set.iUnion fun a => ↑(f a)) Top.top
    k : Infinite ↑(Set.range f)
    h : LT.lt (Cardinal.mk ↑(Set.range f)) (Cardinal.mk β)
    u : ∀ (b : β), Exists fun a => Membership.mem (f a) b := fun b => Eq.mp (Eq.tr …
    ⊢ False
  -/
  let u' : β → range f := fun b => ⟨f (u b).choose, by simp⟩
  have v' : ∀ a, u' ⁻¹' {⟨f a, by simp⟩} ≤ f a := by
    rintro a p m
    simp? [u']  at m says simp only [mem_preimage, mem_singleton_iff, Subtype.mk.injEq, u'] at m
    rw [← m]
    apply fun b => (u b).choose_spec
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Infinite β
    f : α → Finset β
    w : Eq (Set.iUnion fun a => ↑(f a)) Top.top
    k : Infinite ↑(Set.range f)
    h : LT.lt (Cardinal.mk ↑(Set.range f)) (Cardinal.mk β)
    u : ∀ (b : β), Exists fun a => Membership.mem (f a) b := fun b => Eq.mp (Eq.tr …
    u' : β → ↑(Set.range f) := fun b => ⟨f ⋯.choose, ⋯⟩
    v' : ∀ (a : α), LE.le (Set.preimage u' (Singleton.singleton ⟨f a, ⋯⟩)) ↑(f a)
    ⊢ False
  -/
  obtain ⟨⟨-, ⟨a, rfl⟩⟩, p⟩ := exists_infinite_fiber u' h k
  /-
    case intro.mk.intro
    α : Type u_1
    β : Type u_2
    inst✝ : Infinite β
    f : α → Finset β
    w : Eq (Set.iUnion fun a => ↑(f a)) Top.top
    k : Infinite ↑(Set.range f)
    h : LT.lt (Cardinal.mk ↑(Set.range f)) (Cardinal.mk β)
    u : ∀ (b : β), Exists fun a => Membership.mem (f a) b := fun b => Eq.mp (Eq.tr …
    u' : β → ↑(Set.range f) := fun b => ⟨f ⋯.choose, ⋯⟩
    v' : ∀ (a : α), LE.le (Set.preimage u' (Singleton.singleton ⟨f a, ⋯⟩)) ↑(f a)
    a : α
    p : Infinite ↑(Set.preimage u' (Singleton.singleton ⟨f a, ⋯⟩))
    ⊢ False
  -/
  exact (@Infinite.of_injective _ _ p (inclusion (v' a)) (inclusion_injective _)).false
  /-
    🎉 no goals
  -/


theorem lsub_lt_ord_lift_of_isRegular {ι} {f : ι → Ordinal} {c} (hc : IsRegular c)
    (hι : Cardinal.lift.{v, u} #ι < c) : (∀ i, f i < c.ord) → Ordinal.lsub.{u, v} f < c.ord :=
                       /-
                         ι : Type u
                         f : ι → Ordinal.{max u v}
                         c : Cardinal.{max u v}
                         hc : c.IsRegular
                         hι : LT.lt (Cardinal.lift.{v, u} (Cardinal.mk ι)) c
                         ⊢ LT.lt (Cardinal.lift.{v, u} (Cardinal.mk ι)) c.ord.cof
                       -/
  lsub_lt_ord_lift (by rwa [hc.cof_eq])
                       /-
                         🎉 no goals
                       -/


theorem lsub_lt_ord_of_isRegular {ι} {f : ι → Ordinal} {c} (hc : IsRegular c) (hι : #ι < c) :
    (∀ i, f i < c.ord) → Ordinal.lsub f < c.ord :=
                  /-
                    ι : Type (max u_1 u_2)
                    f : ι → Ordinal.{max u_1 u_2}
                    c : Cardinal.{max u_1 u_2}
                    hc : c.IsRegular
                    hι : LT.lt (Cardinal.mk ι) c
                    ⊢ LT.lt (Cardinal.mk ι) c.ord.cof
                  -/
  lsub_lt_ord (by rwa [hc.cof_eq])
                  /-
                    🎉 no goals
                  -/


theorem iSup_lt_ord_lift_of_isRegular {ι} {f : ι → Ordinal} {c} (hc : IsRegular c)
    (hι : Cardinal.lift.{v, u} #ι < c) : (∀ i, f i < c.ord) → iSup f < c.ord :=
                       /-
                         ι : Type u
                         f : ι → Ordinal.{max u v}
                         c : Cardinal.{max u v}
                         hc : c.IsRegular
                         hι : LT.lt (Cardinal.lift.{v, u} (Cardinal.mk ι)) c
                         ⊢ LT.lt (Cardinal.lift.{?u.105483, u} (Cardinal.mk ι)) c.ord.cof
                       -/
  iSup_lt_ord_lift (by rwa [hc.cof_eq])
                       /-
                         🎉 no goals
                       -/


set_option linter.deprecated false in
@[deprecated iSup_lt_ord_lift_of_isRegular (since := "2024-08-27")]
theorem sup_lt_ord_lift_of_isRegular {ι} {f : ι → Ordinal} {c} (hc : IsRegular c)
    (hι : Cardinal.lift.{v, u} #ι < c) : (∀ i, f i < c.ord) → Ordinal.sup.{u, v} f < c.ord :=
  iSup_lt_ord_lift_of_isRegular hc hι


theorem iSup_lt_ord_of_isRegular {ι} {f : ι → Ordinal} {c} (hc : IsRegular c) (hι : #ι < c) :
    (∀ i, f i < c.ord) → iSup f < c.ord :=
                  /-
                    ι : Type u_1
                    f : ι → Ordinal.{u_1}
                    c : Cardinal.{u_1}
                    hc : c.IsRegular
                    hι : LT.lt (Cardinal.mk ι) c
                    ⊢ LT.lt (Cardinal.mk ι) c.ord.cof
                  -/
  iSup_lt_ord (by rwa [hc.cof_eq])
                  /-
                    🎉 no goals
                  -/


set_option linter.deprecated false in
@[deprecated iSup_lt_ord_of_isRegular (since := "2024-08-27")]
theorem sup_lt_ord_of_isRegular {ι} {f : ι → Ordinal} {c} (hc : IsRegular c) (hι : #ι < c) :
    (∀ i, f i < c.ord) → Ordinal.sup f < c.ord :=
  iSup_lt_ord_of_isRegular hc hι


theorem blsub_lt_ord_lift_of_isRegular {o : Ordinal} {f : ∀ a < o, Ordinal} {c} (hc : IsRegular c)
    (ho : Cardinal.lift.{v, u} o.card < c) :
    (∀ i hi, f i hi < c.ord) → Ordinal.blsub.{u, v} o f < c.ord :=
                        /-
                          o : Ordinal.{u}
                          f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
                          c : Cardinal.{max u v}
                          hc : c.IsRegular
                          ho : LT.lt (Cardinal.lift.{v, u} o.card) c
                          ⊢ LT.lt (Cardinal.lift.{v, u} o.card) c.ord.cof
                        -/
  blsub_lt_ord_lift (by rwa [hc.cof_eq])
                        /-
                          🎉 no goals
                        -/


theorem blsub_lt_ord_of_isRegular {o : Ordinal} {f : ∀ a < o, Ordinal} {c} (hc : IsRegular c)
    (ho : o.card < c) : (∀ i hi, f i hi < c.ord) → Ordinal.blsub o f < c.ord :=
                   /-
                     o : Ordinal.{max u_1 u_2}
                     f : (a : Ordinal.{max u_1 u_2}) → LT.lt a o → Ordinal.{max u_1 u_2}
                     c : Cardinal.{max u_1 u_2}
                     hc : c.IsRegular
                     ho : LT.lt o.card c
                     ⊢ LT.lt o.card c.ord.cof
                   -/
  blsub_lt_ord (by rwa [hc.cof_eq])
                   /-
                     🎉 no goals
                   -/


theorem bsup_lt_ord_lift_of_isRegular {o : Ordinal} {f : ∀ a < o, Ordinal} {c} (hc : IsRegular c)
    (hι : Cardinal.lift.{v, u} o.card < c) :
    (∀ i hi, f i hi < c.ord) → Ordinal.bsup.{u, v} o f < c.ord :=
                       /-
                         o : Ordinal.{u}
                         f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
                         c : Cardinal.{max u v}
                         hc : c.IsRegular
                         hι : LT.lt (Cardinal.lift.{v, u} o.card) c
                         ⊢ LT.lt (Cardinal.lift.{v, u} o.card) c.ord.cof
                       -/
  bsup_lt_ord_lift (by rwa [hc.cof_eq])
                       /-
                         🎉 no goals
                       -/


theorem bsup_lt_ord_of_isRegular {o : Ordinal} {f : ∀ a < o, Ordinal} {c} (hc : IsRegular c)
    (hι : o.card < c) : (∀ i hi, f i hi < c.ord) → Ordinal.bsup o f < c.ord :=
                  /-
                    o : Ordinal.{max u_1 u_2}
                    f : (a : Ordinal.{max u_1 u_2}) → LT.lt a o → Ordinal.{max u_1 u_2}
                    c : Cardinal.{max u_1 u_2}
                    hc : c.IsRegular
                    hι : LT.lt o.card c
                    ⊢ LT.lt o.card c.ord.cof
                  -/
  bsup_lt_ord (by rwa [hc.cof_eq])
                  /-
                    🎉 no goals
                  -/


theorem iSup_lt_lift_of_isRegular {ι} {f : ι → Cardinal} {c} (hc : IsRegular c)
    (hι : Cardinal.lift.{v, u} #ι < c) : (∀ i, f i < c) → iSup.{max u v + 1, u + 1} f < c :=
                          /-
                            ι : Type u
                            f : ι → Cardinal.{max u v}
                            c : Cardinal.{max u v}
                            hc : c.IsRegular
                            hι : LT.lt (Cardinal.lift.{v, u} (Cardinal.mk ι)) c
                            ⊢ LT.lt (Cardinal.lift.{v, u} (Cardinal.mk ι)) c.ord.cof
                          -/
  iSup_lt_lift.{u, v} (by rwa [hc.cof_eq])
                          /-
                            🎉 no goals
                          -/


theorem iSup_lt_of_isRegular {ι} {f : ι → Cardinal} {c} (hc : IsRegular c) (hι : #ι < c) :
    (∀ i, f i < c) → iSup f < c :=
              /-
                ι : Type u_1
                f : ι → Cardinal.{u_1}
                c : Cardinal.{u_1}
                hc : c.IsRegular
                hι : LT.lt (Cardinal.mk ι) c
                ⊢ LT.lt (Cardinal.mk ι) c.ord.cof
              -/
  iSup_lt (by rwa [hc.cof_eq])
              /-
                🎉 no goals
              -/


theorem sum_lt_lift_of_isRegular {ι : Type u} {f : ι → Cardinal} {c : Cardinal} (hc : IsRegular c)
    (hι : Cardinal.lift.{v, u} #ι < c) (hf : ∀ i, f i < c) : sum f < c :=
  (sum_le_iSup_lift _).trans_lt <| mul_lt_of_lt hc.1 hι (iSup_lt_lift_of_isRegular hc hι hf)


theorem sum_lt_of_isRegular {ι : Type u} {f : ι → Cardinal} {c : Cardinal} (hc : IsRegular c)
    (hι : #ι < c) : (∀ i, f i < c) → sum f < c :=
                                         /-
                                           ι : Type u
                                           f : ι → Cardinal.{u}
                                           c : Cardinal.{u}
                                           hc : c.IsRegular
                                           hι : LT.lt (Cardinal.mk ι) c
                                           ⊢ LT.lt (Cardinal.lift.{u, u} (Cardinal.mk ι)) c
                                         -/
  sum_lt_lift_of_isRegular.{u, u} hc (by rwa [lift_id])
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
theorem card_lt_of_card_iUnion_lt {ι : Type u} {α : Type u} {t : ι → Set α} {c : Cardinal}
    (h : #(⋃ i, t i) < c) (i : ι) : #(t i) < c :=
  lt_of_le_of_lt (Cardinal.mk_le_mk_of_subset <| subset_iUnion _ _) h


@[simp]
theorem card_iUnion_lt_iff_forall_of_isRegular {ι : Type u} {α : Type u} {t : ι → Set α}
    {c : Cardinal} (hc : c.IsRegular) (hι : #ι < c) : #(⋃ i, t i) < c ↔ ∀ i, #(t i) < c := by
  /-
    ι α : Type u
    t : ι → Set α
    c : Cardinal.{u}
    hc : c.IsRegular
    hι : LT.lt (Cardinal.mk ι) c
    ⊢ Iff (LT.lt (Cardinal.mk ↑(Set.iUnion fun i => t i)) c) (∀ (i : ι), LT.lt (Ca …
  -/
  refine ⟨card_lt_of_card_iUnion_lt, fun h ↦ ?_⟩
  /-
    ι α : Type u
    t : ι → Set α
    c : Cardinal.{u}
    hc : c.IsRegular
    hι : LT.lt (Cardinal.mk ι) c
    h : ∀ (i : ι), LT.lt (Cardinal.mk ↑(t i)) c
    ⊢ LT.lt (Cardinal.mk ↑(Set.iUnion fun i => t i)) c
  -/
  apply lt_of_le_of_lt (Cardinal.mk_sUnion_le _)
  apply Cardinal.mul_lt_of_lt hc.aleph0_le
    (lt_of_le_of_lt Cardinal.mk_range_le hι)
  /-
    ι α : Type u
    t : ι → Set α
    c : Cardinal.{u}
    hc : c.IsRegular
    hι : LT.lt (Cardinal.mk ι) c
    h : ∀ (i : ι), LT.lt (Cardinal.mk ↑(t i)) c
    ⊢ LT.lt (iSup fun s => Cardinal.mk ↑↑s) c
  -/
  apply Cardinal.iSup_lt_of_isRegular hc (lt_of_le_of_lt Cardinal.mk_range_le hι)
  /-
    ι α : Type u
    t : ι → Set α
    c : Cardinal.{u}
    hc : c.IsRegular
    hι : LT.lt (Cardinal.mk ι) c
    h : ∀ (i : ι), LT.lt (Cardinal.mk ↑(t i)) c
    ⊢ ∀ (i : ↑(Set.range fun i => t i)), LT.lt (Cardinal.mk ↑↑i) c
  -/
  simpa
  /-
    🎉 no goals
  -/


theorem card_lt_of_card_biUnion_lt {α β : Type u} {s : Set α} {t : ∀ a ∈ s, Set β} {c : Cardinal}
    (h : #(⋃ a ∈ s, t a ‹_›) < c) (a : α) (ha : a ∈ s) : # (t a ha) < c := by
  /-
    α β : Type u
    s : Set α
    t : (a : α) → Membership.mem s a → Set β
    c : Cardinal.{u}
    h : LT.lt (Cardinal.mk ↑(Set.iUnion fun a => Set.iUnion fun h => t a h)) c
    a : α
    ha : Membership.mem s a
    ⊢ LT.lt (Cardinal.mk ↑(t a ha)) c
  -/
  rw [biUnion_eq_iUnion] at h
  /-
    α β : Type u
    s : Set α
    t : (a : α) → Membership.mem s a → Set β
    c : Cardinal.{u}
    h : LT.lt (Cardinal.mk ↑(Set.iUnion fun x => t ↑x ⋯)) c
    a : α
    ha : Membership.mem s a
    ⊢ LT.lt (Cardinal.mk ↑(t a ha)) c
  -/
  have := card_lt_of_card_iUnion_lt h
  simp_all only [iUnion_coe_set,
    Subtype.forall]


theorem card_biUnion_lt_iff_forall_of_isRegular {α β : Type u} {s : Set α} {t : ∀ a ∈ s, Set β}
    {c : Cardinal} (hc : c.IsRegular) (hs : #s < c) :
    #(⋃ a ∈ s, t a ‹_›) < c ↔ ∀ a (ha : a ∈ s), # (t a ha) < c := by
  /-
    α β : Type u
    s : Set α
    t : (a : α) → Membership.mem s a → Set β
    c : Cardinal.{u}
    hc : c.IsRegular
    hs : LT.lt (Cardinal.mk ↑s) c
    ⊢ Iff (LT.lt (Cardinal.mk ↑(Set.iUnion fun a => Set.iUnion fun h => t a h)) c) …
  -/
  rw [biUnion_eq_iUnion, card_iUnion_lt_iff_forall_of_isRegular hc hs, SetCoe.forall']
  /-
    🎉 no goals
  -/


theorem nfpFamily_lt_ord_lift_of_isRegular {ι} {f : ι → Ordinal → Ordinal} {c} (hc : IsRegular c)
    (hι : Cardinal.lift.{v, u} #ι < c) (hc' : c ≠ ℵ₀) (hf : ∀ (i), ∀ b < c.ord, f i b < c.ord) {a}
    (ha : a < c.ord) : nfpFamily f a < c.ord := by
  /-
    ι : Type u
    f : ι → Ordinal.{max u v} → Ordinal.{max u v}
    c : Cardinal.{max u v}
    hc : c.IsRegular
    hι : LT.lt (Cardinal.lift.{v, u} (Cardinal.mk ι)) c
    hc' : Ne c Cardinal.aleph0
    hf : ∀ (i : ι) (b : Ordinal.{max u v}), LT.lt b c.ord → LT.lt (f i b) c.ord
    a : Ordinal.{max u v}
    ha : LT.lt a c.ord
    ⊢ LT.lt (Ordinal.nfpFamily f a) c.ord
  -/
  apply nfpFamily_lt_ord_lift _ _ hf ha <;> rw [hc.cof_eq]
    /-
      ι : Type u
      f : ι → Ordinal.{max u v} → Ordinal.{max u v}
      c : Cardinal.{max u v}
      hc : c.IsRegular
      hι : LT.lt (Cardinal.lift.{v, u} (Cardinal.mk ι)) c
      hc' : Ne c Cardinal.aleph0
      hf : ∀ (i : ι) (b : Ordinal.{max u v}), LT.lt b c.ord → LT.lt (f i b) c.ord
      a : Ordinal.{max u v}
      ha : LT.lt a c.ord
      ⊢ LT.lt Cardinal.aleph0 c
    -/
  · exact lt_of_le_of_ne hc.1 hc'.symm
    /-
      🎉 no goals
    -/
    /-
      ι : Type u
      f : ι → Ordinal.{max u v} → Ordinal.{max u v}
      c : Cardinal.{max u v}
      hc : c.IsRegular
      hι : LT.lt (Cardinal.lift.{v, u} (Cardinal.mk ι)) c
      hc' : Ne c Cardinal.aleph0
      hf : ∀ (i : ι) (b : Ordinal.{max u v}), LT.lt b c.ord → LT.lt (f i b) c.ord
      a : Ordinal.{max u v}
      ha : LT.lt a c.ord
      ⊢ LT.lt (Cardinal.lift.{v, u} (Cardinal.mk ι)) c
    -/
  · exact hι
    /-
      🎉 no goals
    -/


theorem nfpFamily_lt_ord_of_isRegular {ι} {f : ι → Ordinal → Ordinal} {c} (hc : IsRegular c)
    (hι : #ι < c) (hc' : c ≠ ℵ₀) {a} (hf : ∀ (i), ∀ b < c.ord, f i b < c.ord) :
    a < c.ord → nfpFamily.{u, u} f a < c.ord :=
                                            /-
                                              ι : Type u
                                              f : ι → Ordinal.{u} → Ordinal.{u}
                                              c : Cardinal.{u}
                                              hc : c.IsRegular
                                              hι : LT.lt (Cardinal.mk ι) c
                                              hc' : Ne c Cardinal.aleph0
                                              a : Ordinal.{u}
                                              hf : ∀ (i : ι) (b : Ordinal.{u}), LT.lt b c.ord → LT.lt (f i b) c.ord
                                              ⊢ LT.lt (Cardinal.lift.{?u.136586, u} (Cardinal.mk ι)) c
                                            -/
  nfpFamily_lt_ord_lift_of_isRegular hc (by rwa [lift_id]) hc' hf
                                            /-
                                              🎉 no goals
                                            -/


set_option linter.deprecated false in
@[deprecated nfpFamily_lt_ord_lift_of_isRegular (since := "2024-10-14")]
theorem nfpBFamily_lt_ord_lift_of_isRegular {o : Ordinal} {f : ∀ a < o, Ordinal → Ordinal} {c}
    (hc : IsRegular c) (ho : Cardinal.lift.{v, u} o.card < c) (hc' : c ≠ ℵ₀)
    (hf : ∀ (i hi), ∀ b < c.ord, f i hi b < c.ord) {a} :
    a < c.ord → nfpBFamily.{u, v} o f a < c.ord :=
                                            /-
                                              o : Ordinal.{u}
                                              f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v} → Ordinal.{max u v}
                                              c : Cardinal.{max u v}
                                              hc : c.IsRegular
                                              ho : LT.lt (Cardinal.lift.{v, u} o.card) c
                                              hc' : Ne c Cardinal.aleph0
                                              hf : ∀ (i : Ordinal.{u}) (hi : LT.lt i o) (b : Ordinal.{max u v}), LT.lt b c.o …
                                              a : Ordinal.{max u v}
                                              ⊢ LT.lt (Cardinal.lift.{?u.137264, u} (Cardinal.mk o.toType)) c
                                            -/
  nfpFamily_lt_ord_lift_of_isRegular hc (by rwa [mk_toType]) hc' fun _ => hf _ _
                                            /-
                                              🎉 no goals
                                            -/


set_option linter.deprecated false in
@[deprecated nfpFamily_lt_ord_of_isRegular (since := "2024-10-14")]
theorem nfpBFamily_lt_ord_of_isRegular {o : Ordinal} {f : ∀ a < o, Ordinal → Ordinal} {c}
    (hc : IsRegular c) (ho : o.card < c) (hc' : c ≠ ℵ₀)
    (hf : ∀ (i hi), ∀ b < c.ord, f i hi b < c.ord) {a} :
    a < c.ord → nfpBFamily.{u, u} o f a < c.ord :=
                                             /-
                                               o : Ordinal.{u}
                                               f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{u} → Ordinal.{u}
                                               c : Cardinal.{u}
                                               hc : c.IsRegular
                                               ho : LT.lt o.card c
                                               hc' : Ne c Cardinal.aleph0
                                               hf : ∀ (i : Ordinal.{u}) (hi : LT.lt i o) (b : Ordinal.{u}), LT.lt b c.ord → L …
                                               a : Ordinal.{u}
                                               ⊢ LT.lt (Cardinal.lift.{u, u} o.card) c
                                             -/
  nfpBFamily_lt_ord_lift_of_isRegular hc (by rwa [lift_id]) hc' hf
                                             /-
                                               🎉 no goals
                                             -/


theorem nfp_lt_ord_of_isRegular {f : Ordinal → Ordinal} {c} (hc : IsRegular c) (hc' : c ≠ ℵ₀)
    (hf : ∀ i < c.ord, f i < c.ord) {a} : a < c.ord → nfp f a < c.ord :=
  nfp_lt_ord
    (by
      /-
        f : Ordinal.{u_1} → Ordinal.{u_1}
        c : Cardinal.{u_1}
        hc : c.IsRegular
        hc' : Ne c Cardinal.aleph0
        hf : ∀ (i : Ordinal.{u_1}), LT.lt i c.ord → LT.lt (f i) c.ord
        a : Ordinal.{u_1}
        ⊢ LT.lt Cardinal.aleph0 c.ord.cof
      -/
      rw [hc.cof_eq]
      /-
        f : Ordinal.{u_1} → Ordinal.{u_1}
        c : Cardinal.{u_1}
        hc : c.IsRegular
        hc' : Ne c Cardinal.aleph0
        hf : ∀ (i : Ordinal.{u_1}), LT.lt i c.ord → LT.lt (f i) c.ord
        a : Ordinal.{u_1}
        ⊢ LT.lt Cardinal.aleph0 c
      -/
      exact lt_of_le_of_ne hc.1 hc'.symm)
      /-
        🎉 no goals
      -/
    hf


theorem derivFamily_lt_ord_lift {ι : Type u} {f : ι → Ordinal → Ordinal} {c} (hc : IsRegular c)
    (hι : lift.{v} #ι < c) (hc' : c ≠ ℵ₀) (hf : ∀ i, ∀ b < c.ord, f i b < c.ord) {a} :
    a < c.ord → derivFamily f a < c.ord := by
  have hω : ℵ₀ < c.ord.cof := by
    rw [hc.cof_eq]
    exact lt_of_le_of_ne hc.1 hc'.symm
  induction a using limitRecOn with
  | H₁ =>
    rw [derivFamily_zero]
    exact nfpFamily_lt_ord_lift hω (by rwa [hc.cof_eq]) hf
  | H₂ b hb =>
    intro hb'
    rw [derivFamily_succ]
    exact
      nfpFamily_lt_ord_lift hω (by rwa [hc.cof_eq]) hf
        ((isLimit_ord hc.1).succ_lt (hb ((lt_succ b).trans hb')))
  | H₃ b hb H =>
    intro hb'
    -- TODO: generalize the universes of the lemmas in this file so we don't have to rely on bsup
    have : ⨆ a : Iio b, _ = _ :=
      iSup_eq_bsup.{max u v, max u v} (f := fun x (_ : x < b) ↦ derivFamily f x)
    rw [derivFamily_limit f hb, this]
    exact
      bsup_lt_ord_of_isRegular.{u, v} hc (ord_lt_ord.1 ((ord_card_le b).trans_lt hb')) fun o' ho' =>
        H o' ho' (ho'.trans hb')


theorem derivFamily_lt_ord {ι} {f : ι → Ordinal → Ordinal} {c} (hc : IsRegular c) (hι : #ι < c)
    (hc' : c ≠ ℵ₀) (hf : ∀ (i), ∀ b < c.ord, f i b < c.ord) {a} :
    a < c.ord → derivFamily.{u, u} f a < c.ord :=
                                 /-
                                   ι : Type u
                                   f : ι → Ordinal.{u} → Ordinal.{u}
                                   c : Cardinal.{u}
                                   hc : c.IsRegular
                                   hι : LT.lt (Cardinal.mk ι) c
                                   hc' : Ne c Cardinal.aleph0
                                   hf : ∀ (i : ι) (b : Ordinal.{u}), LT.lt b c.ord → LT.lt (f i b) c.ord
                                   a : Ordinal.{u}
                                   ⊢ LT.lt (Cardinal.lift.{?u.141088, u} (Cardinal.mk ι)) c
                                 -/
  derivFamily_lt_ord_lift hc (by rwa [lift_id]) hc' hf
                                 /-
                                   🎉 no goals
                                 -/


set_option linter.deprecated false in
@[deprecated derivFamily_lt_ord_lift (since := "2024-10-14")]
theorem derivBFamily_lt_ord_lift {o : Ordinal} {f : ∀ a < o, Ordinal → Ordinal} {c}
    (hc : IsRegular c) (hι : Cardinal.lift.{v, u} o.card < c) (hc' : c ≠ ℵ₀)
    (hf : ∀ (i hi), ∀ b < c.ord, f i hi b < c.ord) {a} :
    a < c.ord → derivBFamily.{u, v} o f a < c.ord :=
                                 /-
                                   o : Ordinal.{u}
                                   f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v} → Ordinal.{max u v}
                                   c : Cardinal.{max u v}
                                   hc : c.IsRegular
                                   hι : LT.lt (Cardinal.lift.{v, u} o.card) c
                                   hc' : Ne c Cardinal.aleph0
                                   hf : ∀ (i : Ordinal.{u}) (hi : LT.lt i o) (b : Ordinal.{max u v}), LT.lt b c.o …
                                   a : Ordinal.{max u v}
                                   ⊢ LT.lt (Cardinal.lift.{?u.141766, u} (Cardinal.mk o.toType)) c
                                 -/
  derivFamily_lt_ord_lift hc (by rwa [mk_toType]) hc' fun _ => hf _ _
                                 /-
                                   🎉 no goals
                                 -/


set_option linter.deprecated false in
@[deprecated derivFamily_lt_ord (since := "2024-10-14")]
theorem derivBFamily_lt_ord {o : Ordinal} {f : ∀ a < o, Ordinal → Ordinal} {c} (hc : IsRegular c)
    (hι : o.card < c) (hc' : c ≠ ℵ₀) (hf : ∀ (i hi), ∀ b < c.ord, f i hi b < c.ord) {a} :
    a < c.ord → derivBFamily.{u, u} o f a < c.ord :=
                                  /-
                                    o : Ordinal.{u}
                                    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{u} → Ordinal.{u}
                                    c : Cardinal.{u}
                                    hc : c.IsRegular
                                    hι : LT.lt o.card c
                                    hc' : Ne c Cardinal.aleph0
                                    hf : ∀ (i : Ordinal.{u}) (hi : LT.lt i o) (b : Ordinal.{u}), LT.lt b c.ord → L …
                                    a : Ordinal.{u}
                                    ⊢ LT.lt (Cardinal.lift.{u, u} o.card) c
                                  -/
  derivBFamily_lt_ord_lift hc (by rwa [lift_id]) hc' hf
                                  /-
                                    🎉 no goals
                                  -/


theorem deriv_lt_ord {f : Ordinal.{u} → Ordinal} {c} (hc : IsRegular c) (hc' : c ≠ ℵ₀)
    (hf : ∀ i < c.ord, f i < c.ord) {a} : a < c.ord → deriv f a < c.ord :=
  derivFamily_lt_ord_lift hc
        /-
          f : Ordinal.{u} → Ordinal.{u}
          c : Cardinal.{u}
          hc : c.IsRegular
          hc' : Ne c Cardinal.aleph0
          hf : ∀ (i : Ordinal.{u}), LT.lt i c.ord → LT.lt (f i) c.ord
          a : Ordinal.{u}
          ⊢ LT.lt (Cardinal.lift.{u, 0} (Cardinal.mk Unit)) c
        -/
    (by simpa using Cardinal.one_lt_aleph0.trans (lt_of_le_of_ne hc.1 hc'.symm)) hc' fun _ => hf
        /-
          🎉 no goals
        -/


/-- A cardinal is inaccessible if it is an uncountable regular strong limit cardinal. -/
def IsInaccessible (c : Cardinal) :=
  ℵ₀ < c ∧ IsRegular c ∧ IsStrongLimit c


theorem IsInaccessible.mk {c} (h₁ : ℵ₀ < c) (h₂ : c ≤ c.ord.cof) (h₃ : ∀ x < c, (2^x) < c) :
    IsInaccessible c :=
  ⟨h₁, ⟨h₁.le, h₂⟩, (aleph0_pos.trans h₁).ne', h₃⟩

-- Lean's foundations prove the existence of ℵ₀ many inaccessible cardinals

theorem univ_inaccessible : IsInaccessible univ.{u, v} :=
                        /-
                          ⊢ LT.lt Cardinal.aleph0 Cardinal.univ.{u, v}
                        -/
                        /-
                          🎉 no goals
                        -/
  IsInaccessible.mk (by simpa using lift_lt_univ' ℵ₀) (by simp) fun c h => by
                                                          /-
                                                            🎉 no goals
                                                          -/
    /-
      c : Cardinal.{max (u + 1) v}
      h : LT.lt c Cardinal.univ.{u, v}
      ⊢ LT.lt (HPow.hPow 2 c) Cardinal.univ.{u, v}
    -/
    rcases lt_univ'.1 h with ⟨c, rfl⟩
    /-
      case intro
      c : Cardinal.{u}
      h : LT.lt (Cardinal.lift.{max (u + 1) v, u} c) Cardinal.univ.{u, v}
      ⊢ LT.lt (HPow.hPow 2 (Cardinal.lift.{max (u + 1) v, u} c)) Cardinal.univ.{u, v}
    -/
    rw [← lift_two_power]
    /-
      case intro
      c : Cardinal.{u}
      h : LT.lt (Cardinal.lift.{max (u + 1) v, u} c) Cardinal.univ.{u, v}
      ⊢ LT.lt (Cardinal.lift.{max (u + 1) v, u} (HPow.hPow 2 c)) Cardinal.univ.{u, v}
    -/
    apply lift_lt_univ'
    /-
      🎉 no goals
    -/


theorem lt_power_cof {c : Cardinal.{u}} : ℵ₀ ≤ c → c < (c^cof c.ord) :=
  Cardinal.inductionOn c fun α h => by
    /-
      c : Cardinal.{u}
      α : Type u
      h : LE.le Cardinal.aleph0 (Cardinal.mk α)
      ⊢ LT.lt (Cardinal.mk α) (HPow.hPow (Cardinal.mk α) (Cardinal.mk α).ord.cof)
    -/
    rcases ord_eq α with ⟨r, wo, re⟩
    /-
      case intro.intro
      c : Cardinal.{u}
      α : Type u
      h : LE.le Cardinal.aleph0 (Cardinal.mk α)
      r : α → α → Prop
      wo : IsWellOrder α r
      re : Eq (Cardinal.mk α).ord (Ordinal.type r)
      ⊢ LT.lt (Cardinal.mk α) (HPow.hPow (Cardinal.mk α) (Cardinal.mk α).ord.cof)
    -/
    have := isLimit_ord h
    /-
      case intro.intro
      c : Cardinal.{u}
      α : Type u
      h : LE.le Cardinal.aleph0 (Cardinal.mk α)
      r : α → α → Prop
      wo : IsWellOrder α r
      re : Eq (Cardinal.mk α).ord (Ordinal.type r)
      this : (Cardinal.mk α).ord.IsLimit
      ⊢ LT.lt (Cardinal.mk α) (HPow.hPow (Cardinal.mk α) (Cardinal.mk α).ord.cof)
    -/
    rw [re] at this ⊢
    /-
      case intro.intro
      c : Cardinal.{u}
      α : Type u
      h : LE.le Cardinal.aleph0 (Cardinal.mk α)
      r : α → α → Prop
      wo : IsWellOrder α r
      re : Eq (Cardinal.mk α).ord (Ordinal.type r)
      this : (Ordinal.type r).IsLimit
      ⊢ LT.lt (Cardinal.mk α) (HPow.hPow (Cardinal.mk α) (Ordinal.type r).cof)
    -/
    rcases cof_eq' r this with ⟨S, H, Se⟩
    /-
      case intro.intro.intro.intro
      c : Cardinal.{u}
      α : Type u
      h : LE.le Cardinal.aleph0 (Cardinal.mk α)
      r : α → α → Prop
      wo : IsWellOrder α r
      re : Eq (Cardinal.mk α).ord (Ordinal.type r)
      this : (Ordinal.type r).IsLimit
      S : Set α
      H : ∀ (a : α), Exists fun b => And (Membership.mem S b) (r a b)
      Se : Eq (Cardinal.mk ↑S) (Ordinal.type r).cof
      ⊢ LT.lt (Cardinal.mk α) (HPow.hPow (Cardinal.mk α) (Ordinal.type r).cof)
    -/
    have := sum_lt_prod (fun a : S => #{ x // r x a }) (fun _ => #α) fun i => ?_
      /-
        case intro.intro.intro.intro.refine_2
        c : Cardinal.{u}
        α : Type u
        h : LE.le Cardinal.aleph0 (Cardinal.mk α)
        r : α → α → Prop
        wo : IsWellOrder α r
        re : Eq (Cardinal.mk α).ord (Ordinal.type r)
        this✝ : (Ordinal.type r).IsLimit
        S : Set α
        H : ∀ (a : α), Exists fun b => And (Membership.mem S b) (r a b)
        Se : Eq (Cardinal.mk ↑S) (Ordinal.type r).cof
        this : LT.lt (Cardinal.sum fun a => Cardinal.mk (Subtype fun x => r x ↑a)) (Ca …
        ⊢ LT.lt (Cardinal.mk α) (HPow.hPow (Cardinal.mk α) (Ordinal.type r).cof)
      -/
    · simp only [Cardinal.prod_const, Cardinal.lift_id, ← Se, ← mk_sigma, power_def] at this ⊢
      /-
        case intro.intro.intro.intro.refine_2
        c : Cardinal.{u}
        α : Type u
        h : LE.le Cardinal.aleph0 (Cardinal.mk α)
        r : α → α → Prop
        wo : IsWellOrder α r
        re : Eq (Cardinal.mk α).ord (Ordinal.type r)
        this✝ : (Ordinal.type r).IsLimit
        S : Set α
        H : ∀ (a : α), Exists fun b => And (Membership.mem S b) (r a b)
        Se : Eq (Cardinal.mk ↑S) (Ordinal.type r).cof
        this : LT.lt (Cardinal.mk (Sigma fun i => Subtype fun x => r x ↑i)) (Cardinal. …
        ⊢ LT.lt (Cardinal.mk α) (Cardinal.mk (↑S → α))
      -/
      refine lt_of_le_of_lt ?_ this
      /-
        case intro.intro.intro.intro.refine_2
        c : Cardinal.{u}
        α : Type u
        h : LE.le Cardinal.aleph0 (Cardinal.mk α)
        r : α → α → Prop
        wo : IsWellOrder α r
        re : Eq (Cardinal.mk α).ord (Ordinal.type r)
        this✝ : (Ordinal.type r).IsLimit
        S : Set α
        H : ∀ (a : α), Exists fun b => And (Membership.mem S b) (r a b)
        Se : Eq (Cardinal.mk ↑S) (Ordinal.type r).cof
        this : LT.lt (Cardinal.mk (Sigma fun i => Subtype fun x => r x ↑i)) (Cardinal. …
        ⊢ LE.le (Cardinal.mk α) (Cardinal.mk (Sigma fun i => Subtype fun x => r x ↑i))
      -/
      refine ⟨Embedding.ofSurjective ?_ ?_⟩
        /-
          case intro.intro.intro.intro.refine_2.refine_1
          c : Cardinal.{u}
          α : Type u
          h : LE.le Cardinal.aleph0 (Cardinal.mk α)
          r : α → α → Prop
          wo : IsWellOrder α r
          re : Eq (Cardinal.mk α).ord (Ordinal.type r)
          this✝ : (Ordinal.type r).IsLimit
          S : Set α
          H : ∀ (a : α), Exists fun b => And (Membership.mem S b) (r a b)
          Se : Eq (Cardinal.mk ↑S) (Ordinal.type r).cof
          this : LT.lt (Cardinal.mk (Sigma fun i => Subtype fun x => r x ↑i)) (Cardinal. …
          ⊢ (Sigma fun i => Subtype fun x => r x ↑i) → α
        -/
      · exact fun x => x.2.1
        /-
          🎉 no goals
        -/
      · exact fun a =>
          let ⟨b, h, ab⟩ := H a
          ⟨⟨⟨_, h⟩, _, ab⟩, rfl⟩
      /-
        case intro.intro.intro.intro.refine_1
        c : Cardinal.{u}
        α : Type u
        h : LE.le Cardinal.aleph0 (Cardinal.mk α)
        r : α → α → Prop
        wo : IsWellOrder α r
        re : Eq (Cardinal.mk α).ord (Ordinal.type r)
        this : (Ordinal.type r).IsLimit
        S : Set α
        H : ∀ (a : α), Exists fun b => And (Membership.mem S b) (r a b)
        Se : Eq (Cardinal.mk ↑S) (Ordinal.type r).cof
        i : ↑S
        ⊢ LT.lt ((fun a => Cardinal.mk (Subtype fun x => r x ↑a)) i) ((fun x => Cardin …
      -/
    · have := typein_lt_type r i
      /-
        case intro.intro.intro.intro.refine_1
        c : Cardinal.{u}
        α : Type u
        h : LE.le Cardinal.aleph0 (Cardinal.mk α)
        r : α → α → Prop
        wo : IsWellOrder α r
        re : Eq (Cardinal.mk α).ord (Ordinal.type r)
        this✝ : (Ordinal.type r).IsLimit
        S : Set α
        H : ∀ (a : α), Exists fun b => And (Membership.mem S b) (r a b)
        Se : Eq (Cardinal.mk ↑S) (Ordinal.type r).cof
        i : ↑S
        this : LT.lt ((Ordinal.typein r).toRelEmbedding ↑i) (Ordinal.type r)
        ⊢ LT.lt ((fun a => Cardinal.mk (Subtype fun x => r x ↑a)) i) ((fun x => Cardin …
      -/
      rwa [← re, lt_ord] at this
      /-
        🎉 no goals
      -/


theorem lt_cof_power {a b : Cardinal} (ha : ℵ₀ ≤ a) (b1 : 1 < b) : a < cof (b^a).ord := by
  /-
    a b : Cardinal.{u_1}
    ha : LE.le Cardinal.aleph0 a
    b1 : LT.lt 1 b
    ⊢ LT.lt a (HPow.hPow b a).ord.cof
  -/
  have b0 : b ≠ 0 := (zero_lt_one.trans b1).ne'
  /-
    a b : Cardinal.{u_1}
    ha : LE.le Cardinal.aleph0 a
    b1 : LT.lt 1 b
    b0 : Ne b 0
    ⊢ LT.lt a (HPow.hPow b a).ord.cof
  -/
  apply lt_imp_lt_of_le_imp_le (power_le_power_left <| power_ne_zero a b0)
  /-
    a b : Cardinal.{u_1}
    ha : LE.le Cardinal.aleph0 a
    b1 : LT.lt 1 b
    b0 : Ne b 0
    ⊢ LT.lt (HPow.hPow (HPow.hPow b a) a) (HPow.hPow (HPow.hPow b a) (HPow.hPow b  …
  -/
  rw [← power_mul, mul_eq_self ha]
  /-
    a b : Cardinal.{u_1}
    ha : LE.le Cardinal.aleph0 a
    b1 : LT.lt 1 b
    b0 : Ne b 0
    ⊢ LT.lt (HPow.hPow b a) (HPow.hPow (HPow.hPow b a) (HPow.hPow b a).ord.cof)
  -/
  exact lt_power_cof (ha.trans <| (cantor' _ b1).le)
  /-
    🎉 no goals
  -/


lemma iSup_sequence_lt_omega1 {α : Type u} [Countable α]
    (o : α → Ordinal.{max u v}) (ho : ∀ n, o n < (aleph 1).ord) :
    iSup o < (aleph 1).ord := by
  /-
    α : Type u
    inst✝ : Countable α
    o : α → Ordinal.{max u v}
    ho : ∀ (n : α), LT.lt (o n) (Cardinal.aleph 1).ord
    ⊢ LT.lt (iSup o) (Cardinal.aleph 1).ord
  -/
  apply iSup_lt_ord_lift _ ho
  /-
    α : Type u
    inst✝ : Countable α
    o : α → Ordinal.{max u v}
    ho : ∀ (n : α), LT.lt (o n) (Cardinal.aleph 1).ord
    ⊢ LT.lt (Cardinal.lift.{v, u} (Cardinal.mk α)) (Cardinal.aleph 1).ord.cof
  -/
  rw [Cardinal.isRegular_aleph_one.cof_eq]
  /-
    α : Type u
    inst✝ : Countable α
    o : α → Ordinal.{max u v}
    ho : ∀ (n : α), LT.lt (o n) (Cardinal.aleph 1).ord
    ⊢ LT.lt (Cardinal.lift.{v, u} (Cardinal.mk α)) (Cardinal.aleph 1)
  -/
  exact lt_of_le_of_lt mk_le_aleph0 aleph0_lt_aleph_one
  /-
    🎉 no goals
  -/


