/-- Local notation for a relation -/
local infixl:50 " ≼ " => r


/-- A family of elements of α is directed (with respect to a relation `≼` on α)
  if there is a member of the family `≼`-above any pair in the family. -/
def Directed (f : ι → α) :=
  ∀ x y, ∃ z, f x ≼ f z ∧ f y ≼ f z


/-- A subset of α is directed if there is an element of the set `≼`-above any
  pair of elements in the set. -/
def DirectedOn (s : Set α) :=
  ∀ x ∈ s, ∀ y ∈ s, ∃ z ∈ s, x ≼ z ∧ y ≼ z


theorem directedOn_iff_directed {s} : @DirectedOn α r s ↔ Directed r (Subtype.val : s → α) := by
  /-
    α : Type u
    r : α → α → Prop
    s : Set α
    ⊢ Iff (DirectedOn r s) (Directed r Subtype.val)
  -/
  simp only [DirectedOn, Directed, Subtype.exists, exists_and_left, exists_prop, Subtype.forall]
  /-
    α : Type u
    r : α → α → Prop
    s : Set α
    ⊢ Iff (∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → Exists  …
  -/
  exact forall₂_congr fun x _ => by simp [And.comm, and_assoc]
  /-
    🎉 no goals
  -/


alias ⟨DirectedOn.directed_val, _⟩ := directedOn_iff_directed


theorem directedOn_range {f : ι → α} : Directed r f ↔ DirectedOn r (Set.range f) := by
  /-
    α : Type u
    ι : Sort w
    r : α → α → Prop
    f : ι → α
    ⊢ Iff (Directed r f) (DirectedOn r (Set.range f))
  -/
  simp_rw [Directed, DirectedOn, Set.forall_mem_range, Set.exists_range_iff]
  /-
    🎉 no goals
  -/


protected alias ⟨Directed.directedOn_range, _⟩ := directedOn_range


theorem directedOn_image {s : Set β} {f : β → α} :
    DirectedOn r (f '' s) ↔ DirectedOn (f ⁻¹'o r) s := by
  simp only [DirectedOn, Set.mem_image, exists_exists_and_eq_and, forall_exists_index, and_imp,
    forall_apply_eq_imp_iff₂, Order.Preimage]


theorem DirectedOn.mono' {s : Set α} (hs : DirectedOn r s)
    (h : ∀ ⦃a⦄, a ∈ s → ∀ ⦃b⦄, b ∈ s → r a b → r' a b) : DirectedOn r' s := fun _ hx _ hy =>
  let ⟨z, hz, hxz, hyz⟩ := hs _ hx _ hy
  ⟨z, hz, h hx hz hxz, h hy hz hyz⟩


theorem DirectedOn.mono {s : Set α} (h : DirectedOn r s) (H : ∀ ⦃a b⦄, r a b → r' a b) :
    DirectedOn r' s :=
  h.mono' fun _ _ _ _ h ↦ H h


theorem directed_comp {ι} {f : ι → β} {g : β → α} : Directed r (g ∘ f) ↔ Directed (g ⁻¹'o r) f :=
  Iff.rfl


theorem Directed.mono {s : α → α → Prop} {ι} {f : ι → α} (H : ∀ a b, r a b → s a b)
    (h : Directed r f) : Directed s f := fun a b =>
  let ⟨c, h₁, h₂⟩ := h a b
  ⟨c, H _ _ h₁, H _ _ h₂⟩

-- Porting note: due to some interaction with the local notation, `r` became explicit here in lean3

theorem Directed.mono_comp (r : α → α → Prop) {ι} {rb : β → β → Prop} {g : α → β} {f : ι → α}
    (hg : ∀ ⦃x y⦄, r x y → rb (g x) (g y)) (hf : Directed r f) : Directed rb (g ∘ f) :=
  directed_comp.2 <| hf.mono hg


theorem DirectedOn.mono_comp {r : α → α → Prop} {rb : β → β → Prop} {g : α → β} {s : Set α}
    (hg : ∀ ⦃x y⦄, r x y → rb (g x) (g y)) (hf : DirectedOn r s) : DirectedOn rb (g '' s) :=
  directedOn_image.mpr (hf.mono hg)


/-- A set stable by supremum is `≤`-directed. -/
theorem directedOn_of_sup_mem [SemilatticeSup α] {S : Set α}
    (H : ∀ ⦃i j⦄, i ∈ S → j ∈ S → i ⊔ j ∈ S) : DirectedOn (· ≤ ·) S := fun a ha b hb =>
  ⟨a ⊔ b, H ha hb, le_sup_left, le_sup_right⟩


theorem Directed.extend_bot [Preorder α] [OrderBot α] {e : ι → β} {f : ι → α}
    (hf : Directed (· ≤ ·) f) (he : Function.Injective e) :
    Directed (· ≤ ·) (Function.extend e f ⊥) := by
  /-
    α : Type u
    β : Type v
    ι : Sort w
    inst✝¹ : Preorder α
    inst✝ : OrderBot α
    e : ι → β
    f : ι → α
    hf : Directed (fun x1 x2 => LE.le x1 x2) f
    he : Function.Injective e
    ⊢ Directed (fun x1 x2 => LE.le x1 x2) (Function.extend e f Bot.bot)
  -/
  intro a b
  /-
    α : Type u
    β : Type v
    ι : Sort w
    inst✝¹ : Preorder α
    inst✝ : OrderBot α
    e : ι → β
    f : ι → α
    hf : Directed (fun x1 x2 => LE.le x1 x2) f
    he : Function.Injective e
    a b : β
    ⊢ Exists fun z => And ((fun x1 x2 => LE.le x1 x2) (Function.extend e f Bot.bot …
  -/
  rcases (em (∃ i, e i = a)).symm with (ha | ⟨i, rfl⟩)
    /-
      case inl
      α : Type u
      β : Type v
      ι : Sort w
      inst✝¹ : Preorder α
      inst✝ : OrderBot α
      e : ι → β
      f : ι → α
      hf : Directed (fun x1 x2 => LE.le x1 x2) f
      he : Function.Injective e
      a b : β
      ha : Not (Exists fun i => Eq (e i) a)
      ⊢ Exists fun z => And ((fun x1 x2 => LE.le x1 x2) (Function.extend e f Bot.bot …
    -/
  · use b
    /-
      case h
      α : Type u
      β : Type v
      ι : Sort w
      inst✝¹ : Preorder α
      inst✝ : OrderBot α
      e : ι → β
      f : ι → α
      hf : Directed (fun x1 x2 => LE.le x1 x2) f
      he : Function.Injective e
      a b : β
      ha : Not (Exists fun i => Eq (e i) a)
      ⊢ And ((fun x1 x2 => LE.le x1 x2) (Function.extend e f Bot.bot a) (Function.ex …
    -/
    simp [Function.extend_apply' _ _ _ ha]
    /-
      🎉 no goals
    -/
  /-
    case inr.intro
    α : Type u
    β : Type v
    ι : Sort w
    inst✝¹ : Preorder α
    inst✝ : OrderBot α
    e : ι → β
    f : ι → α
    hf : Directed (fun x1 x2 => LE.le x1 x2) f
    he : Function.Injective e
    b : β
    i : ι
    ⊢ Exists fun z => And ((fun x1 x2 => LE.le x1 x2) (Function.extend e f Bot.bot …
  -/
  rcases (em (∃ i, e i = b)).symm with (hb | ⟨j, rfl⟩)
    /-
      case inr.intro.inl
      α : Type u
      β : Type v
      ι : Sort w
      inst✝¹ : Preorder α
      inst✝ : OrderBot α
      e : ι → β
      f : ι → α
      hf : Directed (fun x1 x2 => LE.le x1 x2) f
      he : Function.Injective e
      b : β
      i : ι
      hb : Not (Exists fun i => Eq (e i) b)
      ⊢ Exists fun z => And ((fun x1 x2 => LE.le x1 x2) (Function.extend e f Bot.bot …
    -/
  · use e i
    /-
      case h
      α : Type u
      β : Type v
      ι : Sort w
      inst✝¹ : Preorder α
      inst✝ : OrderBot α
      e : ι → β
      f : ι → α
      hf : Directed (fun x1 x2 => LE.le x1 x2) f
      he : Function.Injective e
      b : β
      i : ι
      hb : Not (Exists fun i => Eq (e i) b)
      ⊢ And ((fun x1 x2 => LE.le x1 x2) (Function.extend e f Bot.bot (e i)) (Functio …
    -/
    simp [Function.extend_apply' _ _ _ hb]
    /-
      🎉 no goals
    -/
  /-
    case inr.intro.inr.intro
    α : Type u
    β : Type v
    ι : Sort w
    inst✝¹ : Preorder α
    inst✝ : OrderBot α
    e : ι → β
    f : ι → α
    hf : Directed (fun x1 x2 => LE.le x1 x2) f
    he : Function.Injective e
    i j : ι
    ⊢ Exists fun z => And ((fun x1 x2 => LE.le x1 x2) (Function.extend e f Bot.bot …
  -/
  rcases hf i j with ⟨k, hi, hj⟩
  /-
    case inr.intro.inr.intro.intro.intro
    α : Type u
    β : Type v
    ι : Sort w
    inst✝¹ : Preorder α
    inst✝ : OrderBot α
    e : ι → β
    f : ι → α
    hf : Directed (fun x1 x2 => LE.le x1 x2) f
    he : Function.Injective e
    i j k : ι
    hi : LE.le (f i) (f k)
    hj : LE.le (f j) (f k)
    ⊢ Exists fun z => And ((fun x1 x2 => LE.le x1 x2) (Function.extend e f Bot.bot …
  -/
  use e k
  /-
    case h
    α : Type u
    β : Type v
    ι : Sort w
    inst✝¹ : Preorder α
    inst✝ : OrderBot α
    e : ι → β
    f : ι → α
    hf : Directed (fun x1 x2 => LE.le x1 x2) f
    he : Function.Injective e
    i j k : ι
    hi : LE.le (f i) (f k)
    hj : LE.le (f j) (f k)
    ⊢ And ((fun x1 x2 => LE.le x1 x2) (Function.extend e f Bot.bot (e i)) (Functio …
  -/
  simp only [he.extend_apply, *, true_and]
  /-
    🎉 no goals
  -/


/-- A set stable by infimum is `≥`-directed. -/
theorem directedOn_of_inf_mem [SemilatticeInf α] {S : Set α}
    (H : ∀ ⦃i j⦄, i ∈ S → j ∈ S → i ⊓ j ∈ S) : DirectedOn (· ≥ ·) S :=
  directedOn_of_sup_mem (α := αᵒᵈ) H


theorem IsTotal.directed [IsTotal α r] (f : ι → α) : Directed r f := fun i j =>
  Or.casesOn (total_of r (f i) (f j)) (fun h => ⟨j, h, refl _⟩) fun h => ⟨i, refl _, h⟩


/-- `IsDirected α r` states that for any elements `a`, `b` there exists an element `c` such that
`r a c` and `r b c`. -/
class IsDirected (α : Type*) (r : α → α → Prop) : Prop where
  /-- For every pair of elements `a` and `b` there is a `c` such that `r a c` and `r b c` -/
  directed (a b : α) : ∃ c, r a c ∧ r b c


theorem directed_of (r : α → α → Prop) [IsDirected α r] (a b : α) : ∃ c, r a c ∧ r b c :=
  IsDirected.directed _ _


theorem directed_of₃ (r : α → α → Prop) [IsDirected α r] [IsTrans α r] (a b c : α) :
    ∃ d, r a d ∧ r b d ∧ r c d :=
  have ⟨e, hae, hbe⟩ := directed_of r a b
  have ⟨f, hef, hcf⟩ := directed_of r e c
  ⟨f, Trans.trans hae hef, Trans.trans hbe hef, hcf⟩


theorem directed_id [IsDirected α r] : Directed r id := directed_of r


theorem directed_id_iff : Directed r id ↔ IsDirected α r :=
  ⟨fun h => ⟨h⟩, @directed_id _ _⟩


theorem directedOn_univ [IsDirected α r] : DirectedOn r Set.univ := fun a _ b _ =>
  let ⟨c, hc⟩ := directed_of r a b
  ⟨c, trivial, hc⟩


theorem directedOn_univ_iff : DirectedOn r Set.univ ↔ IsDirected α r :=
  ⟨fun h =>
    ⟨fun a b =>
      let ⟨c, _, hc⟩ := h a trivial b trivial
      ⟨c, hc⟩⟩,
    @directedOn_univ _ _⟩

-- see Note [lower instance priority]

instance (priority := 100) IsTotal.to_isDirected [IsTotal α r] : IsDirected α r :=
  directed_id_iff.1 <| IsTotal.directed _


theorem isDirected_mono [IsDirected α r] (h : ∀ ⦃a b⦄, r a b → s a b) : IsDirected α s :=
  ⟨fun a b =>
    let ⟨c, ha, hb⟩ := IsDirected.directed a b
    ⟨c, h ha, h hb⟩⟩


theorem exists_ge_ge [LE α] [IsDirected α (· ≤ ·)] (a b : α) : ∃ c, a ≤ c ∧ b ≤ c :=
  directed_of (· ≤ ·) a b


theorem exists_le_le [LE α] [IsDirected α (· ≥ ·)] (a b : α) : ∃ c, c ≤ a ∧ c ≤ b :=
  directed_of (· ≥ ·) a b


instance OrderDual.isDirected_ge [LE α] [IsDirected α (· ≤ ·)] : IsDirected αᵒᵈ (· ≥ ·) := by
  /-
    α : Type u
    β : Type v
    ι : Sort w
    r r' s : α → α → Prop
    inst✝¹ : LE α
    inst✝ : IsDirected α fun x1 x2 => LE.le x1 x2
    ⊢ IsDirected (OrderDual α) fun x1 x2 => GE.ge x1 x2
  -/
  assumption
  /-
    🎉 no goals
  -/


instance OrderDual.isDirected_le [LE α] [IsDirected α (· ≥ ·)] : IsDirected αᵒᵈ (· ≤ ·) := by
  /-
    α : Type u
    β : Type v
    ι : Sort w
    r r' s : α → α → Prop
    inst✝¹ : LE α
    inst✝ : IsDirected α fun x1 x2 => GE.ge x1 x2
    ⊢ IsDirected (OrderDual α) fun x1 x2 => LE.le x1 x2
  -/
  assumption
  /-
    🎉 no goals
  -/


/-- A monotone function on an upwards-directed type is directed. -/
theorem directed_of_isDirected_le [LE α] [IsDirected α (· ≤ ·)] {f : α → β} {r : β → β → Prop}
    (H : ∀ ⦃i j⦄, i ≤ j → r (f i) (f j)) : Directed r f :=
  directed_id.mono_comp _ H


theorem Monotone.directed_le [Preorder α] [IsDirected α (· ≤ ·)] [Preorder β] {f : α → β} :
    Monotone f → Directed (· ≤ ·) f :=
  directed_of_isDirected_le


theorem Antitone.directed_ge [Preorder α] [IsDirected α (· ≤ ·)] [Preorder β] {f : α → β}
    (hf : Antitone f) : Directed (· ≥ ·) f :=
  directed_of_isDirected_le hf


/-- An antitone function on a downwards-directed type is directed. -/
theorem directed_of_isDirected_ge [LE α] [IsDirected α (· ≥ ·)] {r : β → β → Prop} {f : α → β}
    (hf : ∀ a₁ a₂, a₁ ≤ a₂ → r (f a₂) (f a₁)) : Directed r f :=
  directed_of_isDirected_le (α := αᵒᵈ) fun _ _ ↦ hf _ _


theorem Monotone.directed_ge [Preorder α] [IsDirected α (· ≥ ·)] [Preorder β] {f : α → β}
    (hf : Monotone f) : Directed (· ≥ ·) f :=
  directed_of_isDirected_ge hf


theorem Antitone.directed_le [Preorder α] [IsDirected α (· ≥ ·)] [Preorder β] {f : α → β}
    (hf : Antitone f) : Directed (· ≤ ·) f :=
  directed_of_isDirected_ge hf


protected theorem DirectedOn.insert (h : Reflexive r) (a : α) {s : Set α} (hd : DirectedOn r s)
    (ha : ∀ b ∈ s, ∃ c ∈ s, a ≼ c ∧ b ≼ c) : DirectedOn r (insert a s) := by
  /-
    α : Type u
    r : α → α → Prop
    h : Reflexive r
    a : α
    s : Set α
    hd : DirectedOn r s
    ha : ∀ (b : α), Membership.mem s b → Exists fun c => And (Membership.mem s c)  …
    ⊢ DirectedOn r (Insert.insert a s)
  -/
  rintro x (rfl | hx) y (rfl | hy)
    /-
      case inl.inl
      α : Type u
      r : α → α → Prop
      h : Reflexive r
      s : Set α
      hd : DirectedOn r s
      y : α
      ha : ∀ (b : α), Membership.mem s b → Exists fun c => And (Membership.mem s c)  …
      ⊢ Exists fun z => And (Membership.mem (Insert.insert y s) z) (And (r y z) (r y …
    -/
  · exact ⟨y, Set.mem_insert _ _, h _, h _⟩
    /-
      🎉 no goals
    -/
    /-
      case inl.inr
      α : Type u
      r : α → α → Prop
      h : Reflexive r
      s : Set α
      hd : DirectedOn r s
      x : α
      ha : ∀ (b : α), Membership.mem s b → Exists fun c => And (Membership.mem s c)  …
      y : α
      hy : Membership.mem s y
      ⊢ Exists fun z => And (Membership.mem (Insert.insert x s) z) (And (r x z) (r y …
    -/
  · obtain ⟨w, hws, hwr⟩ := ha y hy
    /-
      case inl.inr.intro.intro
      α : Type u
      r : α → α → Prop
      h : Reflexive r
      s : Set α
      hd : DirectedOn r s
      x : α
      ha : ∀ (b : α), Membership.mem s b → Exists fun c => And (Membership.mem s c)  …
      y : α
      hy : Membership.mem s y
      w : α
      hws : Membership.mem s w
      hwr : And (r x w) (r y w)
      ⊢ Exists fun z => And (Membership.mem (Insert.insert x s) z) (And (r x z) (r y …
    -/
    exact ⟨w, Set.mem_insert_of_mem _ hws, hwr⟩
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u
      r : α → α → Prop
      h : Reflexive r
      s : Set α
      hd : DirectedOn r s
      x : α
      hx : Membership.mem s x
      y : α
      ha : ∀ (b : α), Membership.mem s b → Exists fun c => And (Membership.mem s c)  …
      ⊢ Exists fun z => And (Membership.mem (Insert.insert y s) z) (And (r x z) (r y …
    -/
  · obtain ⟨w, hws, hwr⟩ := ha x hx
    /-
      case inr.inl.intro.intro
      α : Type u
      r : α → α → Prop
      h : Reflexive r
      s : Set α
      hd : DirectedOn r s
      x : α
      hx : Membership.mem s x
      y : α
      ha : ∀ (b : α), Membership.mem s b → Exists fun c => And (Membership.mem s c)  …
      w : α
      hws : Membership.mem s w
      hwr : And (r y w) (r x w)
      ⊢ Exists fun z => And (Membership.mem (Insert.insert y s) z) (And (r x z) (r y …
    -/
    exact ⟨w, Set.mem_insert_of_mem _ hws, hwr.symm⟩
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u
      r : α → α → Prop
      h : Reflexive r
      a : α
      s : Set α
      hd : DirectedOn r s
      ha : ∀ (b : α), Membership.mem s b → Exists fun c => And (Membership.mem s c)  …
      x : α
      hx : Membership.mem s x
      y : α
      hy : Membership.mem s y
      ⊢ Exists fun z => And (Membership.mem (Insert.insert a s) z) (And (r x z) (r y …
    -/
  · obtain ⟨w, hws, hwr⟩ := hd x hx y hy
    /-
      case inr.inr.intro.intro
      α : Type u
      r : α → α → Prop
      h : Reflexive r
      a : α
      s : Set α
      hd : DirectedOn r s
      ha : ∀ (b : α), Membership.mem s b → Exists fun c => And (Membership.mem s c)  …
      x : α
      hx : Membership.mem s x
      y : α
      hy : Membership.mem s y
      w : α
      hws : Membership.mem s w
      hwr : And (r x w) (r y w)
      ⊢ Exists fun z => And (Membership.mem (Insert.insert a s) z) (And (r x z) (r y …
    -/
    exact ⟨w, Set.mem_insert_of_mem _ hws, hwr⟩
    /-
      🎉 no goals
    -/


theorem directedOn_singleton (h : Reflexive r) (a : α) : DirectedOn r ({a} : Set α) :=
  fun x hx _ hy => ⟨x, hx, h _, hx.symm ▸ hy.symm ▸ h _⟩


theorem directedOn_pair (h : Reflexive r) {a b : α} (hab : a ≼ b) : DirectedOn r ({a, b} : Set α) :=
  (directedOn_singleton h _).insert h _ fun c hc => ⟨c, hc, hc.symm ▸ hab, h _⟩


theorem directedOn_pair' (h : Reflexive r) {a b : α} (hab : a ≼ b) :
    DirectedOn r ({b, a} : Set α) := by
  /-
    α : Type u
    r : α → α → Prop
    h : Reflexive r
    a b : α
    hab : r a b
    ⊢ DirectedOn r (Insert.insert b (Singleton.singleton a))
  -/
  rw [Set.pair_comm]
  /-
    α : Type u
    r : α → α → Prop
    h : Reflexive r
    a b : α
    hab : r a b
    ⊢ DirectedOn r (Insert.insert a (Singleton.singleton b))
  -/
  apply directedOn_pair h hab
  /-
    🎉 no goals
  -/


protected theorem IsMin.isBot [IsDirected α (· ≥ ·)] (h : IsMin a) : IsBot a := fun b =>
  let ⟨_, hca, hcb⟩ := exists_le_le a b
  (h hca).trans hcb


protected theorem IsMax.isTop [IsDirected α (· ≤ ·)] (h : IsMax a) : IsTop a :=
  h.toDual.isBot


lemma DirectedOn.is_bot_of_is_min {s : Set α} (hd : DirectedOn (· ≥ ·) s)
    {m} (hm : m ∈ s) (hmin : ∀ a ∈ s, a ≤ m → m ≤ a) : ∀ a ∈ s, m ≤ a := fun a as =>
  let ⟨x, xs, xm, xa⟩ := hd m hm a as
  (hmin x xs xm).trans xa


lemma DirectedOn.is_top_of_is_max {s : Set α} (hd : DirectedOn (· ≤ ·) s)
    {m} (hm : m ∈ s) (hmax : ∀ a ∈ s, m ≤ a → a ≤ m) : ∀ a ∈ s, a ≤ m :=
  @DirectedOn.is_bot_of_is_min αᵒᵈ _ s hd m hm hmax


theorem isTop_or_exists_gt [IsDirected α (· ≤ ·)] (a : α) : IsTop a ∨ ∃ b, a < b :=
  (em (IsMax a)).imp IsMax.isTop not_isMax_iff.mp


theorem isBot_or_exists_lt [IsDirected α (· ≥ ·)] (a : α) : IsBot a ∨ ∃ b, b < a :=
  @isTop_or_exists_gt αᵒᵈ _ _ a


theorem isBot_iff_isMin [IsDirected α (· ≥ ·)] : IsBot a ↔ IsMin a :=
  ⟨IsBot.isMin, IsMin.isBot⟩


theorem isTop_iff_isMax [IsDirected α (· ≤ ·)] : IsTop a ↔ IsMax a :=
  ⟨IsTop.isMax, IsMax.isTop⟩


variable (β) in
theorem exists_lt_of_directed_ge [IsDirected β (· ≥ ·)] :
    ∃ a b : β, a < b := by
  /-
    β : Type v
    inst✝² : PartialOrder β
    inst✝¹ : Nontrivial β
    inst✝ : IsDirected β fun x1 x2 => GE.ge x1 x2
    ⊢ Exists fun a => Exists fun b => LT.lt a b
  -/
  rcases exists_pair_ne β with ⟨a, b, hne⟩
  /-
    case intro.intro
    β : Type v
    inst✝² : PartialOrder β
    inst✝¹ : Nontrivial β
    inst✝ : IsDirected β fun x1 x2 => GE.ge x1 x2
    a b : β
    hne : Ne a b
    ⊢ Exists fun a => Exists fun b => LT.lt a b
  -/
  rcases isBot_or_exists_lt a with (ha | ⟨c, hc⟩)
  /-
    case intro.intro.inl
    β : Type v
    inst✝² : PartialOrder β
    inst✝¹ : Nontrivial β
    inst✝ : IsDirected β fun x1 x2 => GE.ge x1 x2
    a b : β
    hne : Ne a b
    ha : IsBot a
    ⊢ Exists fun a => Exists fun b => LT.lt a b
  -/
  exacts [⟨a, b, (ha b).lt_of_ne hne⟩, ⟨_, _, hc⟩]
  /-
    🎉 no goals
  -/


variable (β) in
theorem exists_lt_of_directed_le [IsDirected β (· ≤ ·)] :
    ∃ a b : β, a < b :=
  let ⟨a, b, h⟩ := exists_lt_of_directed_ge βᵒᵈ
  ⟨b, a, h⟩


protected theorem IsMin.not_isMax [IsDirected β (· ≥ ·)] {b : β} (hb : IsMin b) : ¬ IsMax b := by
  /-
    β : Type v
    inst✝² : PartialOrder β
    inst✝¹ : Nontrivial β
    inst✝ : IsDirected β fun x1 x2 => GE.ge x1 x2
    b : β
    hb : IsMin b
    ⊢ Not (IsMax b)
  -/
  intro hb'
  /-
    β : Type v
    inst✝² : PartialOrder β
    inst✝¹ : Nontrivial β
    inst✝ : IsDirected β fun x1 x2 => GE.ge x1 x2
    b : β
    hb : IsMin b
    hb' : IsMax b
    ⊢ False
  -/
  obtain ⟨a, c, hac⟩ := exists_lt_of_directed_ge β
  /-
    case intro.intro
    β : Type v
    inst✝² : PartialOrder β
    inst✝¹ : Nontrivial β
    inst✝ : IsDirected β fun x1 x2 => GE.ge x1 x2
    b : β
    hb : IsMin b
    hb' : IsMax b
    a c : β
    hac : LT.lt a c
    ⊢ False
  -/
  have := hb.isBot a
  /-
    case intro.intro
    β : Type v
    inst✝² : PartialOrder β
    inst✝¹ : Nontrivial β
    inst✝ : IsDirected β fun x1 x2 => GE.ge x1 x2
    b : β
    hb : IsMin b
    hb' : IsMax b
    a c : β
    hac : LT.lt a c
    this : LE.le b a
    ⊢ False
  -/
  obtain rfl := (hb' <| this).antisymm this
  /-
    case intro.intro
    β : Type v
    inst✝² : PartialOrder β
    inst✝¹ : Nontrivial β
    inst✝ : IsDirected β fun x1 x2 => GE.ge x1 x2
    a c : β
    hac : LT.lt a c
    hb : IsMin a
    hb' : IsMax a
    this : LE.le a a
    ⊢ False
  -/
  exact hb'.not_lt hac
  /-
    🎉 no goals
  -/


protected theorem IsMin.not_isMax' [IsDirected β (· ≤ ·)] {b : β} (hb : IsMin b) : ¬ IsMax b :=
  fun hb' ↦ hb'.toDual.not_isMax hb.toDual


protected theorem IsMax.not_isMin [IsDirected β (· ≤ ·)] {b : β} (hb : IsMax b) : ¬ IsMin b :=
  fun hb' ↦ hb.toDual.not_isMax hb'.toDual


protected theorem IsMax.not_isMin' [IsDirected β (· ≥ ·)] {b : β} (hb : IsMax b) : ¬ IsMin b :=
  fun hb' ↦ hb'.toDual.not_isMin hb.toDual


/-- If `f` is monotone and antitone on a directed order, then `f` is constant. -/
lemma constant_of_monotone_antitone [IsDirected α (· ≤ ·)] (hf : Monotone f) (hf' : Antitone f)
    (a b : α) : f a = f b := by
  /-
    α : Type u
    β : Type v
    inst✝² : PartialOrder β
    inst✝¹ : Preorder α
    f : α → β
    inst✝ : IsDirected α fun x1 x2 => LE.le x1 x2
    hf : Monotone f
    hf' : Antitone f
    a b : α
    ⊢ Eq (f a) (f b)
  -/
  obtain ⟨c, hac, hbc⟩ := exists_ge_ge a b
  /-
    case intro.intro
    α : Type u
    β : Type v
    inst✝² : PartialOrder β
    inst✝¹ : Preorder α
    f : α → β
    inst✝ : IsDirected α fun x1 x2 => LE.le x1 x2
    hf : Monotone f
    hf' : Antitone f
    a b c : α
    hac : LE.le a c
    hbc : LE.le b c
    ⊢ Eq (f a) (f b)
  -/
  exact le_antisymm ((hf hac).trans <| hf' hbc) ((hf hbc).trans <| hf' hac)
  /-
    🎉 no goals
  -/


/-- If `f` is monotone and antitone on a directed set `s`, then `f` is constant on `s`. -/
lemma constant_of_monotoneOn_antitoneOn (hf : MonotoneOn f s) (hf' : AntitoneOn f s)
    (hs : DirectedOn (· ≤ ·) s) : ∀ ⦃a⦄, a ∈ s → ∀ ⦃b⦄, b ∈ s → f a = f b := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PartialOrder β
    inst✝ : Preorder α
    f : α → β
    s : Set α
    hf : MonotoneOn f s
    hf' : AntitoneOn f s
    hs : DirectedOn (fun x1 x2 => LE.le x1 x2) s
    ⊢ ∀ ⦃a : α⦄, Membership.mem s a → ∀ ⦃b : α⦄, Membership.mem s b → Eq (f a) (f b)
  -/
  rintro a ha b hb
  /-
    α : Type u
    β : Type v
    inst✝¹ : PartialOrder β
    inst✝ : Preorder α
    f : α → β
    s : Set α
    hf : MonotoneOn f s
    hf' : AntitoneOn f s
    hs : DirectedOn (fun x1 x2 => LE.le x1 x2) s
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    ⊢ Eq (f a) (f b)
  -/
  obtain ⟨c, hc, hac, hbc⟩ := hs _ ha _ hb
  /-
    case intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : PartialOrder β
    inst✝ : Preorder α
    f : α → β
    s : Set α
    hf : MonotoneOn f s
    hf' : AntitoneOn f s
    hs : DirectedOn (fun x1 x2 => LE.le x1 x2) s
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    c : α
    hc : Membership.mem s c
    hac : LE.le a c
    hbc : LE.le b c
    ⊢ Eq (f a) (f b)
  -/
  exact le_antisymm ((hf ha hc hac).trans <| hf' hb hc hbc) ((hf hb hc hbc).trans <| hf' ha hc hac)
  /-
    🎉 no goals
  -/


instance (priority := 100) SemilatticeSup.to_isDirected_le [SemilatticeSup α] :
    IsDirected α (· ≤ ·) :=
  ⟨fun a b => ⟨a ⊔ b, le_sup_left, le_sup_right⟩⟩

-- see Note [lower instance priority]

instance (priority := 100) SemilatticeInf.to_isDirected_ge [SemilatticeInf α] :
    IsDirected α (· ≥ ·) :=
  ⟨fun a b => ⟨a ⊓ b, inf_le_left, inf_le_right⟩⟩

-- see Note [lower instance priority]

instance (priority := 100) OrderTop.to_isDirected_le [LE α] [OrderTop α] : IsDirected α (· ≤ ·) :=
  ⟨fun _ _ => ⟨⊤, le_top _, le_top _⟩⟩

-- see Note [lower instance priority]

instance (priority := 100) OrderBot.to_isDirected_ge [LE α] [OrderBot α] : IsDirected α (· ≥ ·) :=
  ⟨fun _ _ => ⟨⊥, bot_le _, bot_le _⟩⟩


lemma proj {d : Set (Π i, α i)} (hd : DirectedOn (fun x y => ∀ i, r i (x i) (y i)) d) (i : ι) :
    DirectedOn (r i) ((fun a => a i) '' d) :=
  DirectedOn.mono_comp (fun _ _ h => h) (mono hd fun ⦃_ _⦄ h ↦ h i)


lemma pi {d : (i : ι) → Set (α i)} (hd : ∀ (i : ι), DirectedOn (r i) (d i)) :
    DirectedOn (fun x y => ∀ i, r i (x i) (y i)) (Set.pi Set.univ d) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    r : (i : ι) → α i → α i → Prop
    d : (i : ι) → Set (α i)
    hd : ∀ (i : ι), DirectedOn (r i) (d i)
    ⊢ DirectedOn (fun x y => ∀ (i : ι), r i (x i) (y i)) (Set.univ.pi d)
  -/
  intro a ha b hb
  /-
    ι : Type u_1
    α : ι → Type u_2
    r : (i : ι) → α i → α i → Prop
    d : (i : ι) → Set (α i)
    hd : ∀ (i : ι), DirectedOn (r i) (d i)
    a : (i : ι) → α i
    ha : Membership.mem (Set.univ.pi d) a
    b : (i : ι) → α i
    hb : Membership.mem (Set.univ.pi d) b
    ⊢ Exists fun z => And (Membership.mem (Set.univ.pi d) z) (And ((fun x y => ∀ ( …
  -/
  choose f hfd haf hbf using fun i => hd i (a i) (ha i trivial) (b i) (hb i trivial)
  /-
    ι : Type u_1
    α : ι → Type u_2
    r : (i : ι) → α i → α i → Prop
    d : (i : ι) → Set (α i)
    hd : ∀ (i : ι), DirectedOn (r i) (d i)
    a : (i : ι) → α i
    ha : Membership.mem (Set.univ.pi d) a
    b : (i : ι) → α i
    hb : Membership.mem (Set.univ.pi d) b
    f : (i : ι) → α i
    hfd : ∀ (i : ι), Membership.mem (d i) (f i)
    haf : ∀ (i : ι), r i (a i) (f i)
    hbf : ∀ (i : ι), r i (b i) (f i)
    ⊢ Exists fun z => And (Membership.mem (Set.univ.pi d) z) (And ((fun x y => ∀ ( …
  -/
  exact ⟨f, fun i _ => hfd i, haf, hbf⟩
  /-
    🎉 no goals
  -/


/-- Local notation for a relation -/
local infixl:50 " ≼₁ " => r

/-- Local notation for a relation -/
local infixl:50 " ≼₂ " => r₂


lemma fst {d : Set (α × β)} (hd : DirectedOn (fun p q ↦ p.1 ≼₁ q.1 ∧ p.2 ≼₂ q.2) d) :
    DirectedOn (· ≼₁ ·) (Prod.fst '' d) :=
  DirectedOn.mono_comp (fun ⦃_ _⦄ h ↦ h) (mono hd fun ⦃_ _⦄ h ↦ h.1)


lemma snd {d : Set (α × β)} (hd : DirectedOn (fun p q ↦ p.1 ≼₁ q.1 ∧ p.2 ≼₂ q.2) d) :
    DirectedOn (· ≼₂ ·) (Prod.snd '' d) :=
  DirectedOn.mono_comp (fun ⦃_ _⦄ h ↦ h) (mono hd fun ⦃_ _⦄ h ↦ h.2)


lemma prod {d₁ : Set α} {d₂ : Set β} (h₁ : DirectedOn (· ≼₁ ·) d₁) (h₂ : DirectedOn (· ≼₂ ·) d₂) :
    DirectedOn (fun p q ↦ p.1 ≼₁ q.1 ∧ p.2 ≼₂ q.2) (d₁ ×ˢ d₂) := fun _ hpd _ hqd => by
  /-
    α : Type u
    β : Type v
    r : α → α → Prop
    r₂ : β → β → Prop
    d₁ : Set α
    d₂ : Set β
    h₁ : DirectedOn (fun x1 x2 => r x1 x2) d₁
    h₂ : DirectedOn (fun x1 x2 => r₂ x1 x2) d₂
    x✝¹ : Prod α β
    hpd : Membership.mem (SProd.sprod d₁ d₂) x✝¹
    x✝ : Prod α β
    hqd : Membership.mem (SProd.sprod d₁ d₂) x✝
    ⊢ Exists fun z => And (Membership.mem (SProd.sprod d₁ d₂) z) (And ((fun p q => …
  -/
  obtain ⟨r₁, hdr₁, hpr₁, hqr₁⟩ := h₁ _ hpd.1 _ hqd.1
  /-
    case intro.intro.intro
    α : Type u
    β : Type v
    r : α → α → Prop
    r₂ : β → β → Prop
    d₁ : Set α
    d₂ : Set β
    h₁ : DirectedOn (fun x1 x2 => r x1 x2) d₁
    h₂ : DirectedOn (fun x1 x2 => r₂ x1 x2) d₂
    x✝¹ : Prod α β
    hpd : Membership.mem (SProd.sprod d₁ d₂) x✝¹
    x✝ : Prod α β
    hqd : Membership.mem (SProd.sprod d₁ d₂) x✝
    r₁ : α
    hdr₁ : Membership.mem d₁ r₁
    hpr₁ : r x✝¹.1 r₁
    hqr₁ : r x✝.1 r₁
    ⊢ Exists fun z => And (Membership.mem (SProd.sprod d₁ d₂) z) (And ((fun p q => …
  -/
  obtain ⟨r₂, hdr₂, hpr₂, hqr₂⟩ := h₂ _ hpd.2 _ hqd.2
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u
    β : Type v
    r : α → α → Prop
    r₂✝ : β → β → Prop
    d₁ : Set α
    d₂ : Set β
    h₁ : DirectedOn (fun x1 x2 => r x1 x2) d₁
    h₂ : DirectedOn (fun x1 x2 => r₂✝ x1 x2) d₂
    x✝¹ : Prod α β
    hpd : Membership.mem (SProd.sprod d₁ d₂) x✝¹
    x✝ : Prod α β
    hqd : Membership.mem (SProd.sprod d₁ d₂) x✝
    r₁ : α
    hdr₁ : Membership.mem d₁ r₁
    hpr₁ : r x✝¹.1 r₁
    hqr₁ : r x✝.1 r₁
    r₂ : β
    hdr₂ : Membership.mem d₂ r₂
    hpr₂ : r₂✝ x✝¹.2 r₂
    hqr₂ : r₂✝ x✝.2 r₂
    ⊢ Exists fun z => And (Membership.mem (SProd.sprod d₁ d₂) z) (And ((fun p q => …
  -/
  exact ⟨⟨r₁, r₂⟩, ⟨hdr₁, hdr₂⟩, ⟨hpr₁, hpr₂⟩, ⟨hqr₁, hqr₂⟩⟩
  /-
    🎉 no goals
  -/


