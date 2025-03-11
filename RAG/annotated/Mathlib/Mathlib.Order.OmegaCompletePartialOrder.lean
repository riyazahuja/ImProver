/-- A chain is a monotone sequence.

See the definition on page 114 of [gunter1992]. -/
def Chain (α : Type u) [Preorder α] :=
  ℕ →o α


instance : FunLike (Chain α) ℕ α := inferInstanceAs <| FunLike (ℕ →o α) ℕ α

instance : OrderHomClass (Chain α) ℕ α := inferInstanceAs <| OrderHomClass (ℕ →o α) ℕ α


instance [Inhabited α] : Inhabited (Chain α) :=
  ⟨⟨default, fun _ _ _ => le_rfl⟩⟩


instance : Membership α (Chain α) :=
  ⟨fun (c : ℕ →o α) a => ∃ i, a = c i⟩


instance : LE (Chain α) where le x y := ∀ i, ∃ j, x i ≤ y j


lemma isChain_range : IsChain (· ≤ ·) (Set.range c) := Monotone.isChain_range (OrderHomClass.mono c)


lemma directed : Directed (· ≤ ·) c := directedOn_range.2 c.isChain_range.directedOn


/-- `map` function for `Chain` -/
-- Porting note: `simps` doesn't work with type synonyms
-- @[simps! (config := .asFn)]
def map : Chain β :=
  f.comp c


@[simp] theorem map_coe : ⇑(map c f) = f ∘ c := rfl


theorem mem_map (x : α) : x ∈ c → f x ∈ Chain.map c f :=
  fun ⟨i, h⟩ => ⟨i, h.symm ▸ rfl⟩


theorem exists_of_mem_map {b : β} : b ∈ c.map f → ∃ a, a ∈ c ∧ f a = b :=
  fun ⟨i, h⟩ => ⟨c i, ⟨i, rfl⟩, h.symm⟩


@[simp]
theorem mem_map_iff {b : β} : b ∈ c.map f ↔ ∃ a, a ∈ c ∧ f a = b :=
  ⟨exists_of_mem_map _, fun h => by
    /-
      α : Type u_2
      β : Type u_3
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      c : OmegaCompletePartialOrder.Chain α
      f : OrderHom α β
      b : β
      h : Exists fun a => And (Membership.mem c a) (Eq (f a) b)
      ⊢ Membership.mem (c.map f) b
    -/
    rcases h with ⟨w, h, h'⟩
    /-
      case intro.intro
      α : Type u_2
      β : Type u_3
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      c : OmegaCompletePartialOrder.Chain α
      f : OrderHom α β
      b : β
      w : α
      h : Membership.mem c w
      h' : Eq (f w) b
      ⊢ Membership.mem (c.map f) b
    -/
    subst b
    /-
      case intro.intro
      α : Type u_2
      β : Type u_3
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      c : OmegaCompletePartialOrder.Chain α
      f : OrderHom α β
      w : α
      h : Membership.mem c w
      ⊢ Membership.mem (c.map f) (f w)
    -/
    apply mem_map c _ h⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem map_id : c.map OrderHom.id = c :=
  OrderHom.comp_id _


theorem map_comp : (c.map f).map g = c.map (g.comp f) :=
  rfl


@[mono]
theorem map_le_map {g : α →o β} (h : f ≤ g) : c.map f ≤ c.map g :=
              /-
                α : Type u_2
                β : Type u_3
                inst✝¹ : Preorder α
                inst✝ : Preorder β
                c : OmegaCompletePartialOrder.Chain α
                f g : OrderHom α β
                h : LE.le f g
                i : Nat
                ⊢ Exists fun j => LE.le ((c.map f) i) ((c.map g) j)
              -/
  fun i => by simp only [map_coe, Function.comp_apply]; exists i; apply h
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- `OmegaCompletePartialOrder.Chain.zip` pairs up the elements of two chains
that have the same index. -/
-- Porting note: `simps` doesn't work with type synonyms
-- @[simps!]
def zip (c₀ : Chain α) (c₁ : Chain β) : Chain (α × β) :=
  OrderHom.prod c₀ c₁


@[simp] theorem zip_coe (c₀ : Chain α) (c₁ : Chain β) (n : ℕ) : c₀.zip c₁ n = (c₀ n, c₁ n) := rfl


/-- An example of a `Chain` constructed from an ordered pair. -/
def pair (a b : α) (hab : a ≤ b) : Chain α where
  toFun n := match n with
    | 0 => a
    | _ => b
                        /-
                          ι : Sort u_1
                          α : Type u_2
                          β : Type u_3
                          γ : Type u_4
                          δ : Type u_5
                          inst✝² : Preorder α
                          inst✝¹ : Preorder β
                          inst✝ : Preorder γ
                          c c' : OmegaCompletePartialOrder.Chain α
                          f : OrderHom α β
                          g : OrderHom β γ
                          a b : α
                          hab : LE.le a b
                          x✝² x✝¹ : Nat
                          x✝ : LE.le x✝² x✝¹
                          ⊢ LE.le ((fun n => OmegaCompletePartialOrder.Chain.pair.match_1 (fun n => α) n …
                        -/
  monotone' _ _ _ := by aesop
                        /-
                          🎉 no goals
                        -/


@[simp] lemma pair_zero (a b : α) (hab) : pair a b hab 0 = a := rfl

@[simp] lemma pair_succ (a b : α) (hab) (n : ℕ) : pair a b hab (n + 1) = b := rfl


@[simp] lemma range_pair (a b : α) (hab) : Set.range (pair a b hab) = {a, b} := by
  /-
    α : Type u_2
    inst✝ : Preorder α
    a b : α
    hab : LE.le a b
    ⊢ Eq (Set.range ⇑(OmegaCompletePartialOrder.Chain.pair a b hab)) (Insert.inser …
  -/
  ext; exact Nat.or_exists_add_one.symm.trans (by aesop)
       /-
         🎉 no goals
       -/


@[simp] lemma pair_zip_pair (a₁ a₂ : α) (b₁ b₂ : β) (ha hb) :
    (pair a₁ a₂ ha).zip (pair b₁ b₂ hb) = pair (a₁, b₁) (a₂, b₂) (Prod.le_def.2 ⟨ha, hb⟩) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    a₁ a₂ : α
    b₁ b₂ : β
    ha : LE.le a₁ a₂
    hb : LE.le b₁ b₂
    ⊢ Eq ((OmegaCompletePartialOrder.Chain.pair a₁ a₂ ha).zip (OmegaCompletePartia …
  -/
                                       /-
                                         🎉 no goals
                                       -/
  unfold Chain; ext n : 2; cases n <;> rfl
                                       /-
                                         🎉 no goals
                                       -/


/-- An omega-complete partial order is a partial order with a supremum
operation on increasing sequences indexed by natural numbers (which we
call `ωSup`). In this sense, it is strictly weaker than join complete
semi-lattices as only ω-sized totally ordered sets have a supremum.

See the definition on page 114 of [gunter1992]. -/
class OmegaCompletePartialOrder (α : Type*) extends PartialOrder α where
  /-- The supremum of an increasing sequence -/
  ωSup : Chain α → α
  /-- `ωSup` is an upper bound of the increasing sequence -/
  le_ωSup : ∀ c : Chain α, ∀ i, c i ≤ ωSup c
  /-- `ωSup` is a lower bound of the set of upper bounds of the increasing sequence -/
  ωSup_le : ∀ (c : Chain α) (x), (∀ i, c i ≤ x) → ωSup c ≤ x


/-- Transfer an `OmegaCompletePartialOrder` on `β` to an `OmegaCompletePartialOrder` on `α`
using a strictly monotone function `f : β →o α`, a definition of ωSup and a proof that `f` is
continuous with regard to the provided `ωSup` and the ωCPO on `α`. -/
protected abbrev lift [PartialOrder β] (f : β →o α) (ωSup₀ : Chain β → β)
    (h : ∀ x y, f x ≤ f y → x ≤ y) (h' : ∀ c, f (ωSup₀ c) = ωSup (c.map f)) :
    OmegaCompletePartialOrder β where
  ωSup := ωSup₀
                              /-
                                ι : Sort u_1
                                α : Type u_2
                                β : Type u_3
                                γ : Type u_4
                                δ : Type u_5
                                inst✝¹ : OmegaCompletePartialOrder α
                                inst✝ : PartialOrder β
                                f : OrderHom β α
                                ωSup₀ : OmegaCompletePartialOrder.Chain β → β
                                h : ∀ (x y : β), LE.le (f x) (f y) → LE.le x y
                                h' : ∀ (c : OmegaCompletePartialOrder.Chain β), Eq (f (ωSup₀ c)) (OmegaComplet …
                                c : OmegaCompletePartialOrder.Chain β
                                x : β
                                hx : ∀ (i : Nat), LE.le (c i) x
                                ⊢ LE.le (f (ωSup₀ c)) (f x)
                              -/
                           /-
                             ι : Sort u_1
                             α : Type u_2
                             β : Type u_3
                             γ : Type u_4
                             δ : Type u_5
                             inst✝¹ : OmegaCompletePartialOrder α
                             inst✝ : PartialOrder β
                             f : OrderHom β α
                             ωSup₀ : OmegaCompletePartialOrder.Chain β → β
                             h : ∀ (x y : β), LE.le (f x) (f y) → LE.le x y
                             h' : ∀ (c : OmegaCompletePartialOrder.Chain β), Eq (f (ωSup₀ c)) (OmegaComplet …
                             c : OmegaCompletePartialOrder.Chain β
                             i : Nat
                             ⊢ LE.le (f (c i)) (f (ωSup₀ c))
                           -/
  ωSup_le c x hx := h _ _ (by rw [h']; apply ωSup_le; intro i; apply f.monotone (hx i))
                                    /-
                                      🎉 no goals
                                    -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
  le_ωSup c i := h _ _ (by rw [h']; apply le_ωSup (c.map f))


theorem le_ωSup_of_le {c : Chain α} {x : α} (i : ℕ) (h : x ≤ c i) : x ≤ ωSup c :=
  le_trans h (le_ωSup c _)


theorem ωSup_total {c : Chain α} {x : α} (h : ∀ i, c i ≤ x ∨ x ≤ c i) : ωSup c ≤ x ∨ x ≤ ωSup c :=
  by_cases
    (fun (this : ∀ i, c i ≤ x) => Or.inl (ωSup_le _ _ this))
    (fun (this : ¬∀ i, c i ≤ x) =>
                                 /-
                                   α : Type u_2
                                   inst✝ : OmegaCompletePartialOrder α
                                   c : OmegaCompletePartialOrder.Chain α
                                   x : α
                                   h : ∀ (i : Nat), Or (LE.le (c i) x) (LE.le x (c i))
                                   this : Not (∀ (i : Nat), LE.le (c i) x)
                                   ⊢ Exists fun i => Not (LE.le (c i) x)
                                 -/
      have : ∃ i, ¬c i ≤ x := by simp only [not_forall] at this ⊢; assumption
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
      let ⟨i, hx⟩ := this
      have : x ≤ c i := (h i).resolve_left hx
      Or.inr <| le_ωSup_of_le _ this)


@[mono]
theorem ωSup_le_ωSup_of_le {c₀ c₁ : Chain α} (h : c₀ ≤ c₁) : ωSup c₀ ≤ ωSup c₁ :=
  (ωSup_le _ _) fun i => by
    /-
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      c₀ c₁ : OmegaCompletePartialOrder.Chain α
      h : LE.le c₀ c₁
      i : Nat
      ⊢ LE.le (c₀ i) (OmegaCompletePartialOrder.ωSup c₁)
    -/
    obtain ⟨_, h⟩ := h i
    /-
      case intro
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      c₀ c₁ : OmegaCompletePartialOrder.Chain α
      h✝ : LE.le c₀ c₁
      i w✝ : Nat
      h : LE.le (c₀ i) (c₁ w✝)
      ⊢ LE.le (c₀ i) (OmegaCompletePartialOrder.ωSup c₁)
    -/
    exact le_trans h (le_ωSup _ _)
    /-
      🎉 no goals
    -/


@[simp] theorem ωSup_le_iff {c : Chain α} {x : α} : ωSup c ≤ x ↔ ∀ i, c i ≤ x := by
  /-
    α : Type u_2
    inst✝ : OmegaCompletePartialOrder α
    c : OmegaCompletePartialOrder.Chain α
    x : α
    ⊢ Iff (LE.le (OmegaCompletePartialOrder.ωSup c) x) (∀ (i : Nat), LE.le (c i) x)
  -/
  constructor <;> intros
    /-
      case mp
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      c : OmegaCompletePartialOrder.Chain α
      x : α
      a✝ : LE.le (OmegaCompletePartialOrder.ωSup c) x
      i✝ : Nat
      ⊢ LE.le (c i✝) x
    -/
  · trans ωSup c
      /-
        α : Type u_2
        inst✝ : OmegaCompletePartialOrder α
        c : OmegaCompletePartialOrder.Chain α
        x : α
        a✝ : LE.le (OmegaCompletePartialOrder.ωSup c) x
        i✝ : Nat
        ⊢ LE.le (c i✝) (OmegaCompletePartialOrder.ωSup c)
      -/
    · exact le_ωSup _ _
      /-
        🎉 no goals
      -/
      /-
        α : Type u_2
        inst✝ : OmegaCompletePartialOrder α
        c : OmegaCompletePartialOrder.Chain α
        x : α
        a✝ : LE.le (OmegaCompletePartialOrder.ωSup c) x
        i✝ : Nat
        ⊢ LE.le (OmegaCompletePartialOrder.ωSup c) x
      -/
    · assumption
      /-
        🎉 no goals
      -/
  /-
    case mpr
    α : Type u_2
    inst✝ : OmegaCompletePartialOrder α
    c : OmegaCompletePartialOrder.Chain α
    x : α
    a✝ : ∀ (i : Nat), LE.le (c i) x
    ⊢ LE.le (OmegaCompletePartialOrder.ωSup c) x
  -/
  exact ωSup_le _ _ ‹_›
  /-
    🎉 no goals
  -/


lemma isLUB_range_ωSup (c : Chain α) : IsLUB (Set.range c) (ωSup c) := by
  /-
    α : Type u_2
    inst✝ : OmegaCompletePartialOrder α
    c : OmegaCompletePartialOrder.Chain α
    ⊢ IsLUB (Set.range ⇑c) (OmegaCompletePartialOrder.ωSup c)
  -/
  constructor
  · simp only [upperBounds, Set.mem_range, forall_exists_index, forall_apply_eq_imp_iff,
      Set.mem_setOf_eq]
    /-
      case left
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      c : OmegaCompletePartialOrder.Chain α
      ⊢ ∀ (a : Nat), LE.le (c a) (OmegaCompletePartialOrder.ωSup c)
    -/
    exact fun a ↦ le_ωSup c a
    /-
      🎉 no goals
    -/
  · simp only [lowerBounds, upperBounds, Set.mem_range, forall_exists_index,
      forall_apply_eq_imp_iff, Set.mem_setOf_eq]
    /-
      case right
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      c : OmegaCompletePartialOrder.Chain α
      ⊢ ∀ ⦃a : α⦄, (∀ (a_1 : Nat), LE.le (c a_1) a) → LE.le (OmegaCompletePartialOrd …
    -/
    exact fun ⦃a⦄ a_1 ↦ ωSup_le c a a_1
    /-
      🎉 no goals
    -/


lemma ωSup_eq_of_isLUB {c : Chain α} {a : α} (h : IsLUB (Set.range c) a) : a = ωSup c := by
  /-
    α : Type u_2
    inst✝ : OmegaCompletePartialOrder α
    c : OmegaCompletePartialOrder.Chain α
    a : α
    h : IsLUB (Set.range ⇑c) a
    ⊢ Eq a (OmegaCompletePartialOrder.ωSup c)
  -/
  rw [le_antisymm_iff]
  simp only [IsLUB, IsLeast, upperBounds, lowerBounds, Set.mem_range, forall_exists_index,
    forall_apply_eq_imp_iff, Set.mem_setOf_eq] at h
  /-
    α : Type u_2
    inst✝ : OmegaCompletePartialOrder α
    c : OmegaCompletePartialOrder.Chain α
    a : α
    h : And (∀ (a_1 : Nat), LE.le (c a_1) a) (∀ ⦃a_1 : α⦄, (∀ (a : Nat), LE.le (c  …
    ⊢ And (LE.le a (OmegaCompletePartialOrder.ωSup c)) (LE.le (OmegaCompletePartia …
  -/
  constructor
    /-
      case left
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      c : OmegaCompletePartialOrder.Chain α
      a : α
      h : And (∀ (a_1 : Nat), LE.le (c a_1) a) (∀ ⦃a_1 : α⦄, (∀ (a : Nat), LE.le (c  …
      ⊢ LE.le a (OmegaCompletePartialOrder.ωSup c)
    -/
  · apply h.2
    /-
      case left.a
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      c : OmegaCompletePartialOrder.Chain α
      a : α
      h : And (∀ (a_1 : Nat), LE.le (c a_1) a) (∀ ⦃a_1 : α⦄, (∀ (a : Nat), LE.le (c  …
      ⊢ ∀ (a : Nat), LE.le (c a) (OmegaCompletePartialOrder.ωSup c)
    -/
    exact fun a ↦ le_ωSup c a
    /-
      🎉 no goals
    -/
    /-
      case right
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      c : OmegaCompletePartialOrder.Chain α
      a : α
      h : And (∀ (a_1 : Nat), LE.le (c a_1) a) (∀ ⦃a_1 : α⦄, (∀ (a : Nat), LE.le (c  …
      ⊢ LE.le (OmegaCompletePartialOrder.ωSup c) a
    -/
  · rw [ωSup_le_iff]
    /-
      case right
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      c : OmegaCompletePartialOrder.Chain α
      a : α
      h : And (∀ (a_1 : Nat), LE.le (c a_1) a) (∀ ⦃a_1 : α⦄, (∀ (a : Nat), LE.le (c  …
      ⊢ ∀ (i : Nat), LE.le (c i) a
    -/
    apply h.1
    /-
      🎉 no goals
    -/


/-- A subset `p : α → Prop` of the type closed under `ωSup` induces an
`OmegaCompletePartialOrder` on the subtype `{a : α // p a}`. -/
def subtype {α : Type*} [OmegaCompletePartialOrder α] (p : α → Prop)
    (hp : ∀ c : Chain α, (∀ i ∈ c, p i) → p (ωSup c)) : OmegaCompletePartialOrder (Subtype p) :=
  OmegaCompletePartialOrder.lift (OrderHom.Subtype.val p)
    (fun c => ⟨ωSup _, hp (c.map (OrderHom.Subtype.val p)) fun _ ⟨n, q⟩ => q.symm ▸ (c n).2⟩)
    (fun _ _ h => h) (fun _ => rfl)


/-- A function `f` between `ω`-complete partial orders is `ωScottContinuous` if it is
Scott continuous over chains. -/
def ωScottContinuous (f : α → β) : Prop :=
    ScottContinuousOn (Set.range fun c : Chain α => Set.range c) f


lemma _root_.ScottContinuous.ωScottContinuous (hf : ScottContinuous f) : ωScottContinuous f :=
  hf.scottContinuousOn


lemma ωScottContinuous.monotone (h : ωScottContinuous f) : Monotone f :=
  ScottContinuousOn.monotone _ (fun a b hab => by
    /-
      α : Type u_2
      β : Type u_3
      inst✝¹ : OmegaCompletePartialOrder α
      inst✝ : OmegaCompletePartialOrder β
      f : α → β
      h : OmegaCompletePartialOrder.ωScottContinuous f
      a b : α
      hab : LE.le a b
      ⊢ Membership.mem (Set.range fun c => Set.range ⇑c) (Insert.insert a (Singleton …
    -/
    use pair a b hab; exact range_pair a b hab) h
                      /-
                        🎉 no goals
                      -/


lemma ωScottContinuous.isLUB {c : Chain α} (hf : ωScottContinuous f) :
    IsLUB (Set.range (c.map ⟨f, hf.monotone⟩)) (f (ωSup c)) := by
  simpa [map_coe, OrderHom.coe_mk, Set.range_comp]
    using hf (by simp) (Set.range_nonempty _) (isChain_range c).directedOn (isLUB_range_ωSup c)


lemma ωScottContinuous.id : ωScottContinuous (id : α → α) := ScottContinuousOn.id


lemma ωScottContinuous.map_ωSup (hf : ωScottContinuous f) (c : Chain α) :
    f (ωSup c) = ωSup (c.map ⟨f, hf.monotone⟩) := ωSup_eq_of_isLUB hf.isLUB


/-- `ωScottContinuous f` asserts that `f` is both monotone and distributes over ωSup. -/
lemma ωScottContinuous_iff_monotone_map_ωSup :
    ωScottContinuous f ↔ ∃ hf : Monotone f, ∀ c : Chain α, f (ωSup c) = ωSup (c.map ⟨f, hf⟩) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : OmegaCompletePartialOrder β
    f : α → β
    ⊢ Iff (OmegaCompletePartialOrder.ωScottContinuous f) (Exists fun hf => ∀ (c :  …
  -/
  refine ⟨fun hf ↦ ⟨hf.monotone, hf.map_ωSup⟩, ?_⟩
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : OmegaCompletePartialOrder β
    f : α → β
    ⊢ (Exists fun hf => ∀ (c : OmegaCompletePartialOrder.Chain α), Eq (f (OmegaCom …
  -/
  intro hf _ ⟨c, hc⟩ _ _ _ hda
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : OmegaCompletePartialOrder β
    f : α → β
    hf : Exists fun hf => ∀ (c : OmegaCompletePartialOrder.Chain α), Eq (f (OmegaC …
    d✝ : Set α
    c : OmegaCompletePartialOrder.Chain α
    hc : Eq ((fun c => Set.range ⇑c) c) d✝
    a✝² : d✝.Nonempty
    a✝¹ : DirectedOn (fun x1 x2 => LE.le x1 x2) d✝
    a✝ : α
    hda : IsLUB d✝ a✝
    ⊢ IsLUB (Set.image f d✝) (f a✝)
  -/
  convert isLUB_range_ωSup (c.map { toFun := f, monotone' := hf.1 })
    /-
      case h.e'_3
      α : Type u_2
      β : Type u_3
      inst✝¹ : OmegaCompletePartialOrder α
      inst✝ : OmegaCompletePartialOrder β
      f : α → β
      hf : Exists fun hf => ∀ (c : OmegaCompletePartialOrder.Chain α), Eq (f (OmegaC …
      d✝ : Set α
      c : OmegaCompletePartialOrder.Chain α
      hc : Eq ((fun c => Set.range ⇑c) c) d✝
      a✝² : d✝.Nonempty
      a✝¹ : DirectedOn (fun x1 x2 => LE.le x1 x2) d✝
      a✝ : α
      hda : IsLUB d✝ a✝
      ⊢ Eq (Set.image f d✝) (Set.range ⇑(c.map { toFun := f, monotone' := ⋯ }))
    -/
  · rw [map_coe, OrderHom.coe_mk, ← hc, ← (Set.range_comp f ⇑c)]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_4
      α : Type u_2
      β : Type u_3
      inst✝¹ : OmegaCompletePartialOrder α
      inst✝ : OmegaCompletePartialOrder β
      f : α → β
      hf : Exists fun hf => ∀ (c : OmegaCompletePartialOrder.Chain α), Eq (f (OmegaC …
      d✝ : Set α
      c : OmegaCompletePartialOrder.Chain α
      hc : Eq ((fun c => Set.range ⇑c) c) d✝
      a✝² : d✝.Nonempty
      a✝¹ : DirectedOn (fun x1 x2 => LE.le x1 x2) d✝
      a✝ : α
      hda : IsLUB d✝ a✝
      ⊢ Eq (f a✝) (OmegaCompletePartialOrder.ωSup (c.map { toFun := f, monotone' :=  …
    -/
  · rw [← hc] at hda
    /-
      case h.e'_4
      α : Type u_2
      β : Type u_3
      inst✝¹ : OmegaCompletePartialOrder α
      inst✝ : OmegaCompletePartialOrder β
      f : α → β
      hf : Exists fun hf => ∀ (c : OmegaCompletePartialOrder.Chain α), Eq (f (OmegaC …
      d✝ : Set α
      c : OmegaCompletePartialOrder.Chain α
      hc : Eq ((fun c => Set.range ⇑c) c) d✝
      a✝² : d✝.Nonempty
      a✝¹ : DirectedOn (fun x1 x2 => LE.le x1 x2) d✝
      a✝ : α
      hda : IsLUB ((fun c => Set.range ⇑c) c) a✝
      ⊢ Eq (f a✝) (OmegaCompletePartialOrder.ωSup (c.map { toFun := f, monotone' :=  …
    -/
    rw [← hf.2 c, ωSup_eq_of_isLUB hda]
    /-
      🎉 no goals
    -/


alias ⟨ωScottContinuous.monotone_map_ωSup, ωScottContinuous.of_monotone_map_ωSup⟩ :=
  ωScottContinuous_iff_monotone_map_ωSup

/- A monotone function `f : α →o β` is ωScott continuous if and only if it distributes over ωSup. -/

lemma ωScottContinuous_iff_map_ωSup_of_orderHom {f : α →o β} :
    ωScottContinuous f ↔ ∀ c : Chain α, f (ωSup c) = ωSup (c.map f) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : OmegaCompletePartialOrder β
    f : OrderHom α β
    ⊢ Iff (OmegaCompletePartialOrder.ωScottContinuous ⇑f) (∀ (c : OmegaCompletePar …
  -/
  rw [ωScottContinuous_iff_monotone_map_ωSup]
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : OmegaCompletePartialOrder β
    f : OrderHom α β
    ⊢ Iff (Exists fun hf => ∀ (c : OmegaCompletePartialOrder.Chain α), Eq (f (Omeg …
  -/
  exact exists_prop_of_true f.monotone'
  /-
    🎉 no goals
  -/


alias ⟨ωScottContinuous.map_ωSup_of_orderHom, ωScottContinuous.of_map_ωSup_of_orderHom⟩ :=
  ωScottContinuous_iff_map_ωSup_of_orderHom


lemma ωScottContinuous.comp (hg : ωScottContinuous g) (hf : ωScottContinuous f) :
    ωScottContinuous (g.comp f) :=
  ωScottContinuous.of_monotone_map_ωSup
                                      /-
                                        α : Type u_2
                                        β : Type u_3
                                        γ : Type u_4
                                        inst✝² : OmegaCompletePartialOrder α
                                        inst✝¹ : OmegaCompletePartialOrder β
                                        inst✝ : OmegaCompletePartialOrder γ
                                        f : α → β
                                        g : β → γ
                                        hg : OmegaCompletePartialOrder.ωScottContinuous g
                                        hf : OmegaCompletePartialOrder.ωScottContinuous f
                                        ⊢ ∀ (c : OmegaCompletePartialOrder.Chain α), Eq (Function.comp g f (OmegaCompl …
                                      -/
    ⟨hg.monotone.comp hf.monotone, by simp [hf.map_ωSup, hg.map_ωSup, map_comp]⟩
                                      /-
                                        🎉 no goals
                                      -/


lemma ωScottContinuous.const {x : β} : ωScottContinuous (Function.const α x) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : OmegaCompletePartialOrder β
    x : β
    ⊢ OmegaCompletePartialOrder.ωScottContinuous (Function.const α x)
  -/
  simp [ωScottContinuous, ScottContinuousOn, Set.range_nonempty]
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
/-- A monotone function `f : α →o β` is continuous if it distributes over ωSup.

In order to distinguish it from the (more commonly used) continuity from topology
(see `Mathlib/Topology/Basic.lean`), the present definition is often referred to as
"Scott-continuity" (referring to Dana Scott). It corresponds to continuity
in Scott topological spaces (not defined here). -/
@[deprecated ωScottContinuous (since := "2024-05-29")]
def Continuous (f : α →o β) : Prop :=
  ∀ c : Chain α, f (ωSup c) = ωSup (c.map f)


set_option linter.deprecated false in
/-- `Continuous' f` asserts that `f` is both monotone and continuous. -/
@[deprecated ωScottContinuous (since := "2024-05-29")]
def Continuous' (f : α → β) : Prop :=
  ∃ hf : Monotone f, Continuous ⟨f, hf⟩


@[deprecated ωScottContinuous.isLUB (since := "2024-05-29")]
lemma isLUB_of_scottContinuous {c : Chain α} {f : α → β} (hf : ScottContinuous f) :
    IsLUB (Set.range (Chain.map c ⟨f, (ScottContinuous.monotone hf)⟩)) (f (ωSup c)) :=
  ωScottContinuous.isLUB hf.scottContinuousOn


set_option linter.deprecated false in
@[deprecated ScottContinuous.ωScottContinuous (since := "2024-05-29")]
lemma ScottContinuous.continuous' {f : α → β} (hf : ScottContinuous f) : Continuous' f := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : OmegaCompletePartialOrder β
    f : α → β
    hf : ScottContinuous f
    ⊢ OmegaCompletePartialOrder.Continuous' f
  -/
  constructor
    /-
      case h
      α : Type u_2
      β : Type u_3
      inst✝¹ : OmegaCompletePartialOrder α
      inst✝ : OmegaCompletePartialOrder β
      f : α → β
      hf : ScottContinuous f
      ⊢ OmegaCompletePartialOrder.Continuous { toFun := f, monotone' := ?w }
    -/
  · intro c
    /-
      case h
      α : Type u_2
      β : Type u_3
      inst✝¹ : OmegaCompletePartialOrder α
      inst✝ : OmegaCompletePartialOrder β
      f : α → β
      hf : ScottContinuous f
      c : OmegaCompletePartialOrder.Chain α
      ⊢ Eq ({ toFun := f, monotone' := ?w } (OmegaCompletePartialOrder.ωSup c)) (Ome …
    -/
    rw [← (ωSup_eq_of_isLUB (isLUB_of_scottContinuous hf))]
    /-
      case h
      α : Type u_2
      β : Type u_3
      inst✝¹ : OmegaCompletePartialOrder α
      inst✝ : OmegaCompletePartialOrder β
      f : α → β
      hf : ScottContinuous f
      c : OmegaCompletePartialOrder.Chain α
      ⊢ Eq ({ toFun := f, monotone' := ⋯ } (OmegaCompletePartialOrder.ωSup c)) (f (O …
    -/
    simp only [OrderHom.coe_mk]
    /-
      🎉 no goals
    -/


set_option linter.deprecated false in
@[deprecated ωScottContinuous.monotone (since := "2024-05-29")]
theorem Continuous'.to_monotone {f : α → β} (hf : Continuous' f) : Monotone f :=
  hf.fst


set_option linter.deprecated false in
@[deprecated ωScottContinuous.of_monotone_map_ωSup (since := "2024-05-29")]
theorem Continuous.of_bundled (f : α → β) (hf : Monotone f) (hf' : Continuous ⟨f, hf⟩) :
    Continuous' f :=
  ⟨hf, hf'⟩


set_option linter.deprecated false in
@[deprecated ωScottContinuous.of_monotone_map_ωSup (since := "2024-05-29")]
theorem Continuous.of_bundled' (f : α →o β) (hf' : Continuous f) : Continuous' f :=
  ⟨f.mono, hf'⟩


set_option linter.deprecated false in
@[deprecated ωScottContinuous_iff_monotone_map_ωSup (since := "2024-05-29")]
theorem Continuous'.to_bundled (f : α → β) (hf : Continuous' f) : Continuous ⟨f, hf.to_monotone⟩ :=
  hf.snd


set_option linter.deprecated false in
@[simp, norm_cast, deprecated ωScottContinuous_iff_monotone_map_ωSup (since := "2024-05-29")]
theorem continuous'_coe : ∀ {f : α →o β}, Continuous' f ↔ Continuous f
  | ⟨_, hf⟩ => ⟨fun ⟨_, hc⟩ => hc, fun hc => ⟨hf, hc⟩⟩


set_option linter.deprecated false in
@[deprecated ωScottContinuous.id (since := "2024-05-29")]
                                                            /-
                                                              α : Type u_2
                                                              inst✝ : OmegaCompletePartialOrder α
                                                              ⊢ OmegaCompletePartialOrder.Continuous OrderHom.id
                                                            -/
theorem continuous_id : Continuous (@OrderHom.id α _) := by intro c; rw [c.map_id]; rfl
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


set_option linter.deprecated false in
@[deprecated ωScottContinuous.comp (since := "2024-05-29")]
theorem continuous_comp (hfc : Continuous f) (hgc : Continuous g) : Continuous (g.comp f) := by
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    inst✝² : OmegaCompletePartialOrder α
    inst✝¹ : OmegaCompletePartialOrder β
    inst✝ : OmegaCompletePartialOrder γ
    f : OrderHom α β
    g : OrderHom β γ
    hfc : OmegaCompletePartialOrder.Continuous f
    hgc : OmegaCompletePartialOrder.Continuous g
    ⊢ OmegaCompletePartialOrder.Continuous (g.comp f)
  -/
  dsimp [Continuous] at *; intro
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    inst✝² : OmegaCompletePartialOrder α
    inst✝¹ : OmegaCompletePartialOrder β
    inst✝ : OmegaCompletePartialOrder γ
    f : OrderHom α β
    g : OrderHom β γ
    hfc : ∀ (c : OmegaCompletePartialOrder.Chain α), Eq (f (OmegaCompletePartialOr …
    hgc : ∀ (c : OmegaCompletePartialOrder.Chain β), Eq (g (OmegaCompletePartialOr …
    c✝ : OmegaCompletePartialOrder.Chain α
    ⊢ Eq (g (f (OmegaCompletePartialOrder.ωSup c✝))) (OmegaCompletePartialOrder.ωS …
  -/
  rw [hfc, hgc, Chain.map_comp]
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated ωScottContinuous.id (since := "2024-05-29")]
theorem id_continuous' : Continuous' (@id α) :=
  continuous_id.of_bundled' _


set_option linter.deprecated false in
@[deprecated ωScottContinuous.const (since := "2024-05-29")]
theorem continuous_const (x : β) : Continuous (OrderHom.const α x) := fun c =>
                                  /-
                                    α : Type u_2
                                    β : Type u_3
                                    inst✝¹ : OmegaCompletePartialOrder α
                                    inst✝ : OmegaCompletePartialOrder β
                                    x : β
                                    c : OmegaCompletePartialOrder.Chain α
                                    z : β
                                    ⊢ Iff (LE.le (((OrderHom.const α) x) (OmegaCompletePartialOrder.ωSup c)) z) (L …
                                  -/
  eq_of_forall_ge_iff fun z => by rw [ωSup_le_iff, Chain.map_coe, OrderHom.const_coe_coe]; simp
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


set_option linter.deprecated false in
@[deprecated ωScottContinuous.const (since := "2024-05-29")]
theorem const_continuous' (x : β) : Continuous' (Function.const α x) :=
  Continuous.of_bundled' (OrderHom.const α x) (continuous_const x)


theorem eq_of_chain {c : Chain (Part α)} {a b : α} (ha : some a ∈ c) (hb : some b ∈ c) : a = b := by
  /-
    α : Type u_2
    c : OmegaCompletePartialOrder.Chain (Part α)
    a b : α
    ha : Membership.mem c (Part.some a)
    hb : Membership.mem c (Part.some b)
    ⊢ Eq a b
  -/
  cases' ha with i ha; replace ha := ha.symm
  /-
    case intro
    α : Type u_2
    c : OmegaCompletePartialOrder.Chain (Part α)
    a b : α
    hb : Membership.mem c (Part.some b)
    i : Nat
    ha : Eq (c i) (Part.some a)
    ⊢ Eq a b
  -/
  cases' hb with j hb; replace hb := hb.symm
  /-
    case intro.intro
    α : Type u_2
    c : OmegaCompletePartialOrder.Chain (Part α)
    a b : α
    i : Nat
    ha : Eq (c i) (Part.some a)
    j : Nat
    hb : Eq (c j) (Part.some b)
    ⊢ Eq a b
  -/
  rw [eq_some_iff] at ha hb
  /-
    case intro.intro
    α : Type u_2
    c : OmegaCompletePartialOrder.Chain (Part α)
    a b : α
    i : Nat
    ha : Membership.mem (c i) a
    j : Nat
    hb : Membership.mem (c j) b
    ⊢ Eq a b
  -/
  rcases le_total i j with hij | hji
    /-
      case intro.intro.inl
      α : Type u_2
      c : OmegaCompletePartialOrder.Chain (Part α)
      a b : α
      i : Nat
      ha : Membership.mem (c i) a
      j : Nat
      hb : Membership.mem (c j) b
      hij : LE.le i j
      ⊢ Eq a b
    -/
  · have := c.monotone hij _ ha; apply mem_unique this hb
                                 /-
                                   🎉 no goals
                                 -/
    /-
      case intro.intro.inr
      α : Type u_2
      c : OmegaCompletePartialOrder.Chain (Part α)
      a b : α
      i : Nat
      ha : Membership.mem (c i) a
      j : Nat
      hb : Membership.mem (c j) b
      hji : LE.le j i
      ⊢ Eq a b
    -/
  · have := c.monotone hji _ hb; apply Eq.symm; apply mem_unique this ha
                                                /-
                                                  🎉 no goals
                                                -/
  -- Porting note: Old proof
  -- wlog h : i ≤ j := le_total i j using a b i j, b a j i
  -- rw [eq_some_iff] at ha hb
  -- have := c.monotone h _ ha; apply mem_unique this hb


open Classical in
/-- The (noncomputable) `ωSup` definition for the `ω`-CPO structure on `Part α`. -/
protected noncomputable def ωSup (c : Chain (Part α)) : Part α :=
  if h : ∃ a, some a ∈ c then some (Classical.choose h) else none


theorem ωSup_eq_some {c : Chain (Part α)} {a : α} (h : some a ∈ c) : Part.ωSup c = some a :=
  have : ∃ a, some a ∈ c := ⟨a, h⟩
  have a' : some (Classical.choose this) ∈ c := Classical.choose_spec this
  calc
    Part.ωSup c = some (Classical.choose this) := dif_pos this
    _ = some a := congr_arg _ (eq_of_chain a' h)


theorem ωSup_eq_none {c : Chain (Part α)} (h : ¬∃ a, some a ∈ c) : Part.ωSup c = none :=
  dif_neg h


theorem mem_chain_of_mem_ωSup {c : Chain (Part α)} {a : α} (h : a ∈ Part.ωSup c) : some a ∈ c := by
  /-
    α : Type u_2
    c : OmegaCompletePartialOrder.Chain (Part α)
    a : α
    h : Membership.mem (Part.ωSup c) a
    ⊢ Membership.mem c (Part.some a)
  -/
  simp only [Part.ωSup] at h; split_ifs at h with h_1
    /-
      case pos
      α : Type u_2
      c : OmegaCompletePartialOrder.Chain (Part α)
      a : α
      h_1 : Exists fun a => Membership.mem c (Part.some a)
      h : Membership.mem (Part.some (Classical.choose h_1)) a
      ⊢ Membership.mem c (Part.some a)
    -/
  · have h' := Classical.choose_spec h_1
    /-
      case pos
      α : Type u_2
      c : OmegaCompletePartialOrder.Chain (Part α)
      a : α
      h_1 : Exists fun a => Membership.mem c (Part.some a)
      h : Membership.mem (Part.some (Classical.choose h_1)) a
      h' : Membership.mem c (Part.some (Classical.choose h_1))
      ⊢ Membership.mem c (Part.some a)
    -/
    rw [← eq_some_iff] at h
    /-
      case pos
      α : Type u_2
      c : OmegaCompletePartialOrder.Chain (Part α)
      a : α
      h_1 : Exists fun a => Membership.mem c (Part.some a)
      h : Eq (Part.some (Classical.choose h_1)) (Part.some a)
      h' : Membership.mem c (Part.some (Classical.choose h_1))
      ⊢ Membership.mem c (Part.some a)
    -/
    rw [← h]
    /-
      case pos
      α : Type u_2
      c : OmegaCompletePartialOrder.Chain (Part α)
      a : α
      h_1 : Exists fun a => Membership.mem c (Part.some a)
      h : Eq (Part.some (Classical.choose h_1)) (Part.some a)
      h' : Membership.mem c (Part.some (Classical.choose h_1))
      ⊢ Membership.mem c (Part.some (Classical.choose h_1))
    -/
    exact h'
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_2
      c : OmegaCompletePartialOrder.Chain (Part α)
      a : α
      h_1 : Not (Exists fun a => Membership.mem c (Part.some a))
      h : Membership.mem Part.none a
      ⊢ Membership.mem c (Part.some a)
    -/
  · rcases h with ⟨⟨⟩⟩
    /-
      🎉 no goals
    -/


noncomputable instance omegaCompletePartialOrder :
    OmegaCompletePartialOrder (Part α) where
  ωSup := Part.ωSup
  le_ωSup c i := by
    /-
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      c : OmegaCompletePartialOrder.Chain (Part α)
      i : Nat
      ⊢ LE.le (c i) (Part.ωSup c)
    -/
    intro x hx
    /-
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      c : OmegaCompletePartialOrder.Chain (Part α)
      i : Nat
      x : α
      hx : Membership.mem (c i) x
      ⊢ Membership.mem (Part.ωSup c) x
    -/
    rw [← eq_some_iff] at hx ⊢
    /-
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      c : OmegaCompletePartialOrder.Chain (Part α)
      i : Nat
      x : α
      hx : Eq (c i) (Part.some x)
      ⊢ Eq (Part.ωSup c) (Part.some x)
    -/
    rw [ωSup_eq_some]
    /-
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      c : OmegaCompletePartialOrder.Chain (Part α)
      i : Nat
      x : α
      hx : Eq (c i) (Part.some x)
      ⊢ Membership.mem c (Part.some x)
    -/
    rw [← hx]
    /-
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      c : OmegaCompletePartialOrder.Chain (Part α)
      i : Nat
      x : α
      hx : Eq (c i) (Part.some x)
      ⊢ Membership.mem c (c i)
    -/
    exact ⟨i, rfl⟩
    /-
      🎉 no goals
    -/
  ωSup_le := by
    /-
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      ⊢ ∀ (c : OmegaCompletePartialOrder.Chain (Part α)) (x : Part α), (∀ (i : Nat), …
    -/
    rintro c x hx a ha
    /-
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      c : OmegaCompletePartialOrder.Chain (Part α)
      x : Part α
      hx : ∀ (i : Nat), LE.le (c i) x
      a : α
      ha : Membership.mem (Part.ωSup c) a
      ⊢ Membership.mem x a
    -/
    replace ha := mem_chain_of_mem_ωSup ha
    /-
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      c : OmegaCompletePartialOrder.Chain (Part α)
      x : Part α
      hx : ∀ (i : Nat), LE.le (c i) x
      a : α
      ha : Membership.mem c (Part.some a)
      ⊢ Membership.mem x a
    -/
    cases' ha with i ha
    /-
      case intro
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      c : OmegaCompletePartialOrder.Chain (Part α)
      x : Part α
      hx : ∀ (i : Nat), LE.le (c i) x
      a : α
      i : Nat
      ha : Eq (Part.some a) (c i)
      ⊢ Membership.mem x a
    -/
    apply hx i
    /-
      case intro.a
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      c : OmegaCompletePartialOrder.Chain (Part α)
      x : Part α
      hx : ∀ (i : Nat), LE.le (c i) x
      a : α
      i : Nat
      ha : Eq (Part.some a) (c i)
      ⊢ Membership.mem (c i) a
    -/
    rw [← ha]
    /-
      case intro.a
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      c : OmegaCompletePartialOrder.Chain (Part α)
      x : Part α
      hx : ∀ (i : Nat), LE.le (c i) x
      a : α
      i : Nat
      ha : Eq (Part.some a) (c i)
      ⊢ Membership.mem (Part.some a) a
    -/
    apply mem_some
    /-
      🎉 no goals
    -/


theorem mem_ωSup (x : α) (c : Chain (Part α)) : x ∈ ωSup c ↔ some x ∈ c := by
  /-
    α : Type u_2
    x : α
    c : OmegaCompletePartialOrder.Chain (Part α)
    ⊢ Iff (Membership.mem (OmegaCompletePartialOrder.ωSup c) x) (Membership.mem c  …
  -/
  simp only [ωSup, Part.ωSup]
  /-
    α : Type u_2
    x : α
    c : OmegaCompletePartialOrder.Chain (Part α)
    ⊢ Iff (Membership.mem (dite (Exists fun a => Membership.mem c (Part.some a)) ( …
  -/
  constructor
    /-
      case mp
      α : Type u_2
      x : α
      c : OmegaCompletePartialOrder.Chain (Part α)
      ⊢ Membership.mem (dite (Exists fun a => Membership.mem c (Part.some a)) (fun h …
    -/
  · split_ifs with h
    /-
      case pos
      α : Type u_2
      x : α
      c : OmegaCompletePartialOrder.Chain (Part α)
      h : Exists fun a => Membership.mem c (Part.some a)
      ⊢ Membership.mem (Part.some (Classical.choose h)) x → Membership.mem c (Part.s …
    -/
    swap
      /-
        case neg
        α : Type u_2
        x : α
        c : OmegaCompletePartialOrder.Chain (Part α)
        h : Not (Exists fun a => Membership.mem c (Part.some a))
        ⊢ Membership.mem Part.none x → Membership.mem c (Part.some x)
      -/
    · rintro ⟨⟨⟩⟩
      /-
        🎉 no goals
      -/
    /-
      case pos
      α : Type u_2
      x : α
      c : OmegaCompletePartialOrder.Chain (Part α)
      h : Exists fun a => Membership.mem c (Part.some a)
      ⊢ Membership.mem (Part.some (Classical.choose h)) x → Membership.mem c (Part.s …
    -/
    intro h'
    /-
      case pos
      α : Type u_2
      x : α
      c : OmegaCompletePartialOrder.Chain (Part α)
      h : Exists fun a => Membership.mem c (Part.some a)
      h' : Membership.mem (Part.some (Classical.choose h)) x
      ⊢ Membership.mem c (Part.some x)
    -/
    have hh := Classical.choose_spec h
    /-
      case pos
      α : Type u_2
      x : α
      c : OmegaCompletePartialOrder.Chain (Part α)
      h : Exists fun a => Membership.mem c (Part.some a)
      h' : Membership.mem (Part.some (Classical.choose h)) x
      hh : Membership.mem c (Part.some (Classical.choose h))
      ⊢ Membership.mem c (Part.some x)
    -/
    simp only [mem_some_iff] at h'
    /-
      case pos
      α : Type u_2
      x : α
      c : OmegaCompletePartialOrder.Chain (Part α)
      h : Exists fun a => Membership.mem c (Part.some a)
      hh : Membership.mem c (Part.some (Classical.choose h))
      h' : Eq x (Classical.choose h)
      ⊢ Membership.mem c (Part.some x)
    -/
    subst x
    /-
      case pos
      α : Type u_2
      c : OmegaCompletePartialOrder.Chain (Part α)
      h : Exists fun a => Membership.mem c (Part.some a)
      hh : Membership.mem c (Part.some (Classical.choose h))
      ⊢ Membership.mem c (Part.some (Classical.choose h))
    -/
    exact hh
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_2
      x : α
      c : OmegaCompletePartialOrder.Chain (Part α)
      ⊢ Membership.mem c (Part.some x) → Membership.mem (dite (Exists fun a => Membe …
    -/
  · intro h
    /-
      case mpr
      α : Type u_2
      x : α
      c : OmegaCompletePartialOrder.Chain (Part α)
      h : Membership.mem c (Part.some x)
      ⊢ Membership.mem (dite (Exists fun a => Membership.mem c (Part.some a)) (fun h …
    -/
    have h' : ∃ a : α, some a ∈ c := ⟨_, h⟩
    /-
      case mpr
      α : Type u_2
      x : α
      c : OmegaCompletePartialOrder.Chain (Part α)
      h : Membership.mem c (Part.some x)
      h' : Exists fun a => Membership.mem c (Part.some a)
      ⊢ Membership.mem (dite (Exists fun a => Membership.mem c (Part.some a)) (fun h …
    -/
    rw [dif_pos h']
    /-
      case mpr
      α : Type u_2
      x : α
      c : OmegaCompletePartialOrder.Chain (Part α)
      h : Membership.mem c (Part.some x)
      h' : Exists fun a => Membership.mem c (Part.some a)
      ⊢ Membership.mem (Part.some (Classical.choose h')) x
    -/
    have hh := Classical.choose_spec h'
    /-
      case mpr
      α : Type u_2
      x : α
      c : OmegaCompletePartialOrder.Chain (Part α)
      h : Membership.mem c (Part.some x)
      h' : Exists fun a => Membership.mem c (Part.some a)
      hh : Membership.mem c (Part.some (Classical.choose h'))
      ⊢ Membership.mem (Part.some (Classical.choose h')) x
    -/
    rw [eq_of_chain hh h]
    /-
      case mpr
      α : Type u_2
      x : α
      c : OmegaCompletePartialOrder.Chain (Part α)
      h : Membership.mem c (Part.some x)
      h' : Exists fun a => Membership.mem c (Part.some a)
      hh : Membership.mem c (Part.some (Classical.choose h'))
      ⊢ Membership.mem (Part.some x) x
    -/
    simp
    /-
      🎉 no goals
    -/


instance [∀ a, OmegaCompletePartialOrder (β a)] :
    OmegaCompletePartialOrder (∀ a, β a) where
  ωSup c a := ωSup (c.map (Pi.evalOrderHom a))
  ωSup_le _ _ hf a :=
    ωSup_le _ _ <| by
      /-
        ι : Sort u_1
        α : Type u_2
        β✝ : Type u_3
        γ : Type u_4
        δ : Type u_5
        β : α → Type u_6
        inst✝ : (a : α) → OmegaCompletePartialOrder (β a)
        x✝¹ : OmegaCompletePartialOrder.Chain ((a : α) → β a)
        x✝ : (a : α) → β a
        hf : ∀ (i : Nat), LE.le (x✝¹ i) x✝
        a : α
        ⊢ ∀ (i : Nat), LE.le ((x✝¹.map (Pi.evalOrderHom a)) i) (x✝ a)
      -/
      rintro i
      /-
        ι : Sort u_1
        α : Type u_2
        β✝ : Type u_3
        γ : Type u_4
        δ : Type u_5
        β : α → Type u_6
        inst✝ : (a : α) → OmegaCompletePartialOrder (β a)
        x✝¹ : OmegaCompletePartialOrder.Chain ((a : α) → β a)
        x✝ : (a : α) → β a
        hf : ∀ (i : Nat), LE.le (x✝¹ i) x✝
        a : α
        i : Nat
        ⊢ LE.le ((x✝¹.map (Pi.evalOrderHom a)) i) (x✝ a)
      -/
      apply hf
      /-
        🎉 no goals
      -/
  le_ωSup _ _ _ := le_ωSup_of_le _ <| le_rfl


lemma ωScottContinuous.apply₂ (hf : ωScottContinuous f) (a : α) : ωScottContinuous (f · a) :=
  ωScottContinuous.of_monotone_map_ωSup
    ⟨fun _ _ h ↦ hf.monotone h a, fun c ↦ congr_fun (hf.map_ωSup c) a⟩


lemma ωScottContinuous.of_apply₂ (hf : ∀ a, ωScottContinuous (f · a)) : ωScottContinuous f :=
  ωScottContinuous.of_monotone_map_ωSup
                                                 /-
                                                   α : Type u_2
                                                   γ : Type u_4
                                                   β : α → Type u_6
                                                   inst✝¹ : (x : α) → OmegaCompletePartialOrder (β x)
                                                   inst✝ : OmegaCompletePartialOrder γ
                                                   f : γ → (x : α) → β x
                                                   hf : ∀ (a : α), OmegaCompletePartialOrder.ωScottContinuous fun x => f x a
                                                   c : OmegaCompletePartialOrder.Chain γ
                                                   ⊢ Eq (f (OmegaCompletePartialOrder.ωSup c)) (OmegaCompletePartialOrder.ωSup (c …
                                                 -/
    ⟨fun _ _ h a ↦ (hf a).monotone h, fun c ↦ by ext a; apply (hf a).map_ωSup c⟩
                                                        /-
                                                          🎉 no goals
                                                        -/


lemma ωScottContinuous_iff_apply₂ : ωScottContinuous f ↔ ∀ a, ωScottContinuous (f · a) :=
  ⟨ωScottContinuous.apply₂, ωScottContinuous.of_apply₂⟩


set_option linter.deprecated false in
@[deprecated ωScottContinuous.apply₂ (since := "2024-05-29")]
theorem flip₁_continuous' (f : ∀ x : α, γ → β x) (a : α) (hf : Continuous' fun x y => f y x) :
    Continuous' (f a) :=
  Continuous.of_bundled _ (fun _ _ h => hf.to_monotone h a) fun c => congr_fun (hf.to_bundled _ c) a


set_option linter.deprecated false in
@[deprecated ωScottContinuous.of_apply₂ (since := "2024-05-29")]
theorem flip₂_continuous' (f : γ → ∀ x, β x) (hf : ∀ x, Continuous' fun g => f g x) :
    Continuous' f :=
  Continuous.of_bundled _ (fun _ _ h a => (hf a).to_monotone h)
        /-
          α : Type u_2
          γ : Type u_4
          β : α → Type u_6
          inst✝¹ : (x : α) → OmegaCompletePartialOrder (β x)
          inst✝ : OmegaCompletePartialOrder γ
          f : γ → (x : α) → β x
          hf : ∀ (x : α), OmegaCompletePartialOrder.Continuous' fun g => f g x
          ⊢ OmegaCompletePartialOrder.Continuous { toFun := f, monotone' := ⋯ }
        -/
    (by intro c; ext a; apply (hf a).to_bundled _ c)
                        /-
                          🎉 no goals
                        -/


/-- The supremum of a chain in the product `ω`-CPO. -/
@[simps]
protected def ωSup (c : Chain (α × β)) : α × β :=
  (ωSup (c.map OrderHom.fst), ωSup (c.map OrderHom.snd))


@[simps! ωSup_fst ωSup_snd]
instance : OmegaCompletePartialOrder (α × β) where
  ωSup := Prod.ωSup
  ωSup_le := fun _ _ h => ⟨ωSup_le _ _ fun i => (h i).1, ωSup_le _ _ fun i => (h i).2⟩
  le_ωSup c i := ⟨le_ωSup (c.map OrderHom.fst) i, le_ωSup (c.map OrderHom.snd) i⟩


theorem ωSup_zip (c₀ : Chain α) (c₁ : Chain β) : ωSup (c₀.zip c₁) = (ωSup c₀, ωSup c₁) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : OmegaCompletePartialOrder β
    c₀ : OmegaCompletePartialOrder.Chain α
    c₁ : OmegaCompletePartialOrder.Chain β
    ⊢ Eq (OmegaCompletePartialOrder.ωSup (c₀.zip c₁)) { fst := OmegaCompletePartia …
  -/
  apply eq_of_forall_ge_iff; rintro ⟨z₁, z₂⟩
  /-
    case H.mk
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : OmegaCompletePartialOrder β
    c₀ : OmegaCompletePartialOrder.Chain α
    c₁ : OmegaCompletePartialOrder.Chain β
    z₁ : α
    z₂ : β
    ⊢ Iff (LE.le (OmegaCompletePartialOrder.ωSup (c₀.zip c₁)) { fst := z₁, snd :=  …
  -/
  simp [ωSup_le_iff, forall_and]
  /-
    🎉 no goals
  -/


/-- Any complete lattice has an `ω`-CPO structure where the countable supremum is a special case
of arbitrary suprema. -/
instance (priority := 100) [CompleteLattice α] : OmegaCompletePartialOrder α where
  ωSup c := ⨆ i, c i
  ωSup_le := fun ⟨c, _⟩ s hs => by
    /-
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝ : CompleteLattice α
      x✝ : OmegaCompletePartialOrder.Chain α
      s : α
      c : Nat → α
      monotone'✝ : Monotone c
      hs : ∀ (i : Nat), LE.le ({ toFun := c, monotone' := monotone'✝ } i) s
      ⊢ LE.le ((fun c => iSup fun i => c i) { toFun := c, monotone' := monotone'✝ }) s
    -/
                                /-
                                  ι : Sort u_1
                                  α : Type u_2
                                  β : Type u_3
                                  γ : Type u_4
                                  δ : Type u_5
                                  inst✝ : CompleteLattice α
                                  x✝ : OmegaCompletePartialOrder.Chain α
                                  i : Nat
                                  c : Nat → α
                                  monotone'✝ : Monotone c
                                  ⊢ LE.le ({ toFun := c, monotone' := monotone'✝ } i) ((fun c => iSup fun i => c …
                                -/
    simp only [iSup_le_iff, OrderHom.coe_mk] at hs ⊢; intro i; apply hs i
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
  le_ωSup := fun ⟨c, _⟩ i => by simp only [OrderHom.coe_mk]; apply le_iSup_of_le i; rfl


open Chain in
lemma ωScottContinuous.prodMk (hf : ωScottContinuous f) (hg : ωScottContinuous g) :
    ωScottContinuous fun x => (f x, g x) := ScottContinuousOn.prodMk (fun a b hab => by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : CompleteLattice β
    f g : α → β
    hf : OmegaCompletePartialOrder.ωScottContinuous f
    hg : OmegaCompletePartialOrder.ωScottContinuous g
    a b : α
    hab : LE.le a b
    ⊢ Membership.mem (Set.range fun c => Set.range ⇑c) (Insert.insert a (Singleton …
  -/
  use pair a b hab; exact range_pair a b hab) hf hg
                    /-
                      🎉 no goals
                    -/


lemma ωScottContinuous.iSup {f : ι → α → β} (hf : ∀ i, ωScottContinuous (f i)) :
    ωScottContinuous (⨆ i, f i) := by
  refine ωScottContinuous.of_monotone_map_ωSup
    ⟨Monotone.iSup fun i ↦ (hf i).monotone, fun c ↦ eq_of_forall_ge_iff fun a ↦ ?_⟩
  /-
    ι : Sort u_1
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : CompleteLattice β
    f : ι → α → β
    hf : ∀ (i : ι), OmegaCompletePartialOrder.ωScottContinuous (f i)
    c : OmegaCompletePartialOrder.Chain α
    a : β
    ⊢ Iff (LE.le (_root_.iSup (fun i => f i) (OmegaCompletePartialOrder.ωSup c)) a …
  -/
  simp +contextual [ωSup_le_iff, (hf _).map_ωSup, @forall_swap ι]
  /-
    🎉 no goals
  -/


lemma ωScottContinuous.sSup {s : Set (α → β)} (hs : ∀ f ∈ s, ωScottContinuous f) :
    ωScottContinuous (sSup s) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : CompleteLattice β
    s : Set (α → β)
    hs : ∀ (f : α → β), Membership.mem s f → OmegaCompletePartialOrder.ωScottConti …
    ⊢ OmegaCompletePartialOrder.ωScottContinuous (SupSet.sSup s)
  -/
  rw [sSup_eq_iSup]; exact ωScottContinuous.iSup fun f ↦ ωScottContinuous.iSup <| hs f
                     /-
                       🎉 no goals
                     -/


lemma ωScottContinuous.sup (hf : ωScottContinuous f) (hg : ωScottContinuous g) :
    ωScottContinuous (f ⊔ g) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : CompleteLattice β
    f g : α → β
    hf : OmegaCompletePartialOrder.ωScottContinuous f
    hg : OmegaCompletePartialOrder.ωScottContinuous g
    ⊢ OmegaCompletePartialOrder.ωScottContinuous (Max.max f g)
  -/
  rw [← sSup_pair]
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : CompleteLattice β
    f g : α → β
    hf : OmegaCompletePartialOrder.ωScottContinuous f
    hg : OmegaCompletePartialOrder.ωScottContinuous g
    ⊢ OmegaCompletePartialOrder.ωScottContinuous (SupSet.sSup (Insert.insert f (Si …
  -/
  apply ωScottContinuous.sSup
  /-
    case hs
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : CompleteLattice β
    f g : α → β
    hf : OmegaCompletePartialOrder.ωScottContinuous f
    hg : OmegaCompletePartialOrder.ωScottContinuous g
    ⊢ ∀ (f_1 : α → β), Membership.mem (Insert.insert f (Singleton.singleton g)) f_ …
  -/
                               /-
                                 🎉 no goals
                               -/
  rintro f (rfl | rfl | _) <;> assumption
                               /-
                                 🎉 no goals
                               -/


lemma ωScottContinuous.top : ωScottContinuous (⊤ : α → β) :=
  ωScottContinuous.of_monotone_map_ωSup
                                                            /-
                                                              α : Type u_2
                                                              β : Type u_3
                                                              inst✝¹ : OmegaCompletePartialOrder α
                                                              inst✝ : CompleteLattice β
                                                              c : OmegaCompletePartialOrder.Chain α
                                                              a : β
                                                              ⊢ Iff (LE.le (Top.top (OmegaCompletePartialOrder.ωSup c)) a) (LE.le (OmegaComp …
                                                            -/
    ⟨monotone_const, fun c ↦ eq_of_forall_ge_iff fun a ↦ by simp⟩
                                                            /-
                                                              🎉 no goals
                                                            -/


lemma ωScottContinuous.bot : ωScottContinuous (⊥ : α → β) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : CompleteLattice β
    ⊢ OmegaCompletePartialOrder.ωScottContinuous Bot.bot
  -/
  rw [← sSup_empty]; exact ωScottContinuous.sSup (by simp)
                     /-
                       🎉 no goals
                     -/


set_option linter.deprecated false in
@[deprecated ωScottContinuous.sSup (since := "2024-05-29")]
theorem sSup_continuous (s : Set <| α →o β) (hs : ∀ f ∈ s, Continuous f) : Continuous (sSup s) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : CompleteLattice β
    s : Set (OrderHom α β)
    hs : ∀ (f : OrderHom α β), Membership.mem s f → OmegaCompletePartialOrder.Cont …
    ⊢ OmegaCompletePartialOrder.Continuous (SupSet.sSup s)
  -/
  intro c
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : CompleteLattice β
    s : Set (OrderHom α β)
    hs : ∀ (f : OrderHom α β), Membership.mem s f → OmegaCompletePartialOrder.Cont …
    c : OmegaCompletePartialOrder.Chain α
    ⊢ Eq ((SupSet.sSup s) (OmegaCompletePartialOrder.ωSup c)) (OmegaCompletePartia …
  -/
  apply eq_of_forall_ge_iff
  /-
    case H
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : CompleteLattice β
    s : Set (OrderHom α β)
    hs : ∀ (f : OrderHom α β), Membership.mem s f → OmegaCompletePartialOrder.Cont …
    c : OmegaCompletePartialOrder.Chain α
    ⊢ ∀ (c_1 : β), Iff (LE.le ((SupSet.sSup s) (OmegaCompletePartialOrder.ωSup c)) …
  -/
  intro z
  suffices (∀ f ∈ s, ∀ n, f (c n) ≤ z) ↔ ∀ n, ∀ f ∈ s, f (c n) ≤ z by
    simpa (config := { contextual := true }) [ωSup_le_iff, hs _ _ _] using this
  /-
    case H
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : CompleteLattice β
    s : Set (OrderHom α β)
    hs : ∀ (f : OrderHom α β), Membership.mem s f → OmegaCompletePartialOrder.Cont …
    c : OmegaCompletePartialOrder.Chain α
    z : β
    ⊢ Iff (∀ (f : OrderHom α β), Membership.mem s f → ∀ (n : Nat), LE.le (f (c n)) …
  -/
  exact ⟨fun H n f hf => H f hf n, fun H f hf n => H n f hf⟩
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated ωScottContinuous.iSup (since := "2024-05-29")]
theorem iSup_continuous {ι : Sort*} {f : ι → α →o β} (h : ∀ i, Continuous (f i)) :
    Continuous (⨆ i, f i) :=
  sSup_continuous _ <| Set.forall_mem_range.2 h


set_option linter.deprecated false in
@[deprecated ωScottContinuous.sSup (since := "2024-05-29")]
theorem sSup_continuous' (s : Set (α → β)) (hc : ∀ f ∈ s, Continuous' f) :
    Continuous' (sSup s) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : CompleteLattice β
    s : Set (α → β)
    hc : ∀ (f : α → β), Membership.mem s f → OmegaCompletePartialOrder.Continuous' f
    ⊢ OmegaCompletePartialOrder.Continuous' (SupSet.sSup s)
  -/
  lift s to Set (α →o β) using fun f hf => (hc f hf).to_monotone
  /-
    case intro
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : CompleteLattice β
    s : Set (OrderHom α β)
    hc : ∀ (f : α → β), Membership.mem (Set.image DFunLike.coe s) f → OmegaComplet …
    ⊢ OmegaCompletePartialOrder.Continuous' (SupSet.sSup (Set.image DFunLike.coe s))
  -/
  simp only [Set.forall_mem_image, continuous'_coe] at hc
  /-
    case intro
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : CompleteLattice β
    s : Set (OrderHom α β)
    hc : ∀ ⦃x : OrderHom α β⦄, Membership.mem s x → OmegaCompletePartialOrder.Cont …
    ⊢ OmegaCompletePartialOrder.Continuous' (SupSet.sSup (Set.image DFunLike.coe s))
  -/
  rw [sSup_image]
  /-
    case intro
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : CompleteLattice β
    s : Set (OrderHom α β)
    hc : ∀ ⦃x : OrderHom α β⦄, Membership.mem s x → OmegaCompletePartialOrder.Cont …
    ⊢ OmegaCompletePartialOrder.Continuous' (iSup fun a => iSup fun h => ⇑a)
  -/
  norm_cast
  /-
    case intro
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : CompleteLattice β
    s : Set (OrderHom α β)
    hc : ∀ ⦃x : OrderHom α β⦄, Membership.mem s x → OmegaCompletePartialOrder.Cont …
    ⊢ OmegaCompletePartialOrder.Continuous (iSup fun i => iSup fun i_1 => i)
  -/
  exact iSup_continuous fun f ↦ iSup_continuous fun hf ↦ hc hf
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated ωScottContinuous.sup (since := "2024-05-29")]
theorem sup_continuous {f g : α →o β} (hf : Continuous f) (hg : Continuous g) :
    Continuous (f ⊔ g) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : CompleteLattice β
    f g : OrderHom α β
    hf : OmegaCompletePartialOrder.Continuous f
    hg : OmegaCompletePartialOrder.Continuous g
    ⊢ OmegaCompletePartialOrder.Continuous (Max.max f g)
  -/
  rw [← sSup_pair]; apply sSup_continuous
  /-
    case hs
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : CompleteLattice β
    f g : OrderHom α β
    hf : OmegaCompletePartialOrder.Continuous f
    hg : OmegaCompletePartialOrder.Continuous g
    ⊢ ∀ (f_1 : OrderHom α β), Membership.mem (Insert.insert f (Singleton.singleton …
  -/
                               /-
                                 🎉 no goals
                               -/
  rintro f (rfl | rfl | _) <;> assumption
                               /-
                                 🎉 no goals
                               -/


set_option linter.deprecated false in
@[deprecated ωScottContinuous.top (since := "2024-05-29")]
theorem top_continuous : Continuous (⊤ : α →o β) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : CompleteLattice β
    ⊢ OmegaCompletePartialOrder.Continuous Top.top
  -/
  intro c; apply eq_of_forall_ge_iff; intro z
  simp only [OrderHom.instTopOrderHom_top, OrderHom.const_coe_coe, Function.const, top_le_iff,
    ωSup_le_iff, Chain.map_coe, Function.comp, forall_const]


set_option linter.deprecated false in
@[deprecated ωScottContinuous.bot (since := "2024-05-29")]
theorem bot_continuous : Continuous (⊥ : α →o β) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : CompleteLattice β
    ⊢ OmegaCompletePartialOrder.Continuous Bot.bot
  -/
  rw [← sSup_empty]
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : CompleteLattice β
    ⊢ OmegaCompletePartialOrder.Continuous (SupSet.sSup EmptyCollection.emptyColle …
  -/
  exact sSup_continuous _ fun f hf => hf.elim
  /-
    🎉 no goals
  -/


lemma ωScottContinuous.inf (hf : ωScottContinuous f) (hg : ωScottContinuous g) :
    ωScottContinuous (f ⊓ g) := by
  refine ωScottContinuous.of_monotone_map_ωSup
    ⟨hf.monotone.inf hg.monotone, fun c ↦ eq_of_forall_ge_iff fun a ↦ ?_⟩
  simp only [Pi.inf_apply, hf.map_ωSup c, hg.map_ωSup c, inf_le_iff, ωSup_le_iff, Chain.map_coe,
    Function.comp, OrderHom.coe_mk, ← forall_or_left, ← forall_or_right]
  exact ⟨fun h _ ↦ h _ _, fun h i j ↦
    (h (max j i)).imp (le_trans <| hf.monotone <| c.mono <| le_max_left _ _)
      (le_trans <| hg.monotone <| c.mono <| le_max_right _ _)⟩


set_option linter.deprecated false in
@[deprecated ωScottContinuous.inf (since := "2024-05-29")]
theorem inf_continuous (f g : α →o β) (hf : Continuous f) (hg : Continuous g) :
    Continuous (f ⊓ g) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : CompleteLinearOrder β
    f g : OrderHom α β
    hf : OmegaCompletePartialOrder.Continuous f
    hg : OmegaCompletePartialOrder.Continuous g
    ⊢ OmegaCompletePartialOrder.Continuous (Min.min f g)
  -/
  refine fun c => eq_of_forall_ge_iff fun z => ?_
  simp only [inf_le_iff, hf c, hg c, ωSup_le_iff, ← forall_or_left, ← forall_or_right,
             Chain.map_coe, OrderHom.coe_inf, Pi.inf_apply, Function.comp]
  exact ⟨fun h _ ↦ h _ _, fun h i j ↦
    (h (max j i)).imp (le_trans <| f.mono <| c.mono <| le_max_left _ _)
      (le_trans <| g.mono <| c.mono <| le_max_right _ _)⟩


set_option linter.deprecated false in
@[deprecated ωScottContinuous.inf (since := "2024-05-29")]
theorem inf_continuous' {f g : α → β} (hf : Continuous' f) (hg : Continuous' g) :
    Continuous' (f ⊓ g) :=
  ⟨_, inf_continuous _ _ hf.snd hg.snd⟩


/-- The `ωSup` operator for monotone functions. -/
@[simps]
protected def ωSup (c : Chain (α →o β)) : α →o β where
  toFun a := ωSup (c.map (OrderHom.apply a))
  monotone' _ _ h := ωSup_le_ωSup_of_le ((Chain.map_le_map _) fun a => a.monotone h)


@[simps! ωSup_coe]
instance omegaCompletePartialOrder : OmegaCompletePartialOrder (α →o β) :=
  OmegaCompletePartialOrder.lift OrderHom.coeFnHom OrderHom.ωSup (fun _ _ h => h) fun _ => rfl


variable (α β) in
/-- A monotone function on `ω`-continuous partial orders is said to be continuous
if for every chain `c : chain α`, `f (⊔ i, c i) = ⊔ i, f (c i)`.
This is just the bundled version of `OrderHom.continuous`. -/
structure ContinuousHom extends OrderHom α β where
  /-- The underlying function of a `ContinuousHom` is continuous, i.e. it preserves `ωSup` -/
  protected map_ωSup' (c : Chain α) : toFun (ωSup c) = ωSup (c.map toOrderHom)


@[inherit_doc] infixr:25 " →𝒄 " => ContinuousHom -- Input: \r\MIc


instance : FunLike (α →𝒄 β) α β where
  coe f := f.toFun
                       /-
                         ι : Sort u_1
                         α : Type u_2
                         β : Type u_3
                         γ : Type u_4
                         δ : Type u_5
                         inst✝³ : OmegaCompletePartialOrder α
                         inst✝² : OmegaCompletePartialOrder β
                         inst✝¹ : OmegaCompletePartialOrder γ
                         inst✝ : OmegaCompletePartialOrder δ
                         ⊢ Function.Injective fun f => f.toFun
                       -/
  coe_injective' := by rintro ⟨⟩ ⟨⟩ h; congr; exact DFunLike.ext' h
                                              /-
                                                🎉 no goals
                                              -/


instance : OrderHomClass (α →𝒄 β) α β where
  map_rel f _ _ h := f.mono h

-- Porting note: removed to avoid conflict with the generic instance
-- instance : Coe (α →𝒄 β) (α →o β) where coe := ContinuousHom.toOrderHom


instance : PartialOrder (α →𝒄 β) :=
                                                        /-
                                                          ι : Sort u_1
                                                          α : Type u_2
                                                          β : Type u_3
                                                          γ : Type u_4
                                                          δ : Type u_5
                                                          inst✝³ : OmegaCompletePartialOrder α
                                                          inst✝² : OmegaCompletePartialOrder β
                                                          inst✝¹ : OmegaCompletePartialOrder γ
                                                          inst✝ : OmegaCompletePartialOrder δ
                                                          ⊢ Function.Injective fun f => f.toFun
                                                        -/
  (PartialOrder.lift fun f => f.toOrderHom.toFun) <| by rintro ⟨⟨⟩⟩ ⟨⟨⟩⟩ h; congr
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


protected lemma ωScottContinuous (f : α →𝒄 β) : ωScottContinuous f :=
  ωScottContinuous.of_map_ωSup_of_orderHom f.map_ωSup'

-- Not a `simp` lemma because in many cases projection is simpler than a generic coercion

theorem toOrderHom_eq_coe (f : α →𝒄 β) : f.1 = f := rfl


@[simp] theorem coe_mk (f : α →o β) (hf) : ⇑(mk f hf) = f := rfl


@[simp] theorem coe_toOrderHom (f : α →𝒄 β) : ⇑f.1 = f := rfl


/-- See Note [custom simps projection]. We specify this explicitly because we don't have a DFunLike
instance.
-/
def Simps.apply (h : α →𝒄 β) : α → β :=
  h


protected theorem congr_fun {f g : α →𝒄 β} (h : f = g) (x : α) : f x = g x :=
  DFunLike.congr_fun h x


protected theorem congr_arg (f : α →𝒄 β) {x y : α} (h : x = y) : f x = f y :=
  congr_arg f h


protected theorem monotone (f : α →𝒄 β) : Monotone f :=
  f.monotone'


@[mono]
theorem apply_mono {f g : α →𝒄 β} {x y : α} (h₁ : f ≤ g) (h₂ : x ≤ y) : f x ≤ g y :=
  OrderHom.apply_mono (show (f : α →o β) ≤ g from h₁) h₂


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided." (since := "2024-07-27")]
theorem ite_continuous' {p : Prop} [hp : Decidable p] (f g : α → β) (hf : Continuous' f)
    (hg : Continuous' g) : Continuous' fun x => if p then f x else g x := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : OmegaCompletePartialOrder β
    p : Prop
    hp : Decidable p
    f g : α → β
    hf : OmegaCompletePartialOrder.Continuous' f
    hg : OmegaCompletePartialOrder.Continuous' g
    ⊢ OmegaCompletePartialOrder.Continuous' fun x => ite p (f x) (g x)
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp [*]
                /-
                  🎉 no goals
                -/


theorem ωSup_bind {β γ : Type v} (c : Chain α) (f : α →o Part β) (g : α →o β → Part γ) :
    ωSup (c.map (f.partBind g)) = ωSup (c.map f) >>= ωSup (c.map g) := by
  /-
    α : Type u_2
    inst✝ : OmegaCompletePartialOrder α
    β γ : Type v
    c : OmegaCompletePartialOrder.Chain α
    f : OrderHom α (Part β)
    g : OrderHom α (β → Part γ)
    ⊢ Eq (OmegaCompletePartialOrder.ωSup (c.map (f.partBind g))) (Bind.bind (Omega …
  -/
  apply eq_of_forall_ge_iff; intro x
  simp only [ωSup_le_iff, Part.bind_le, Chain.mem_map_iff, and_imp, OrderHom.partBind_coe,
    exists_imp]
  /-
    case H
    α : Type u_2
    inst✝ : OmegaCompletePartialOrder α
    β γ : Type v
    c : OmegaCompletePartialOrder.Chain α
    f : OrderHom α (Part β)
    g : OrderHom α (β → Part γ)
    x : Part γ
    ⊢ Iff (∀ (i : Nat), LE.le ((c.map (f.partBind g)) i) x) (∀ (a : β), Membership …
  -/
  constructor <;> intro h'''
    /-
      case H.mp
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      β γ : Type v
      c : OmegaCompletePartialOrder.Chain α
      f : OrderHom α (Part β)
      g : OrderHom α (β → Part γ)
      x : Part γ
      h''' : ∀ (i : Nat), LE.le ((c.map (f.partBind g)) i) x
      ⊢ ∀ (a : β), Membership.mem (OmegaCompletePartialOrder.ωSup (c.map f)) a → LE. …
    -/
  · intro b hb
    /-
      case H.mp
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      β γ : Type v
      c : OmegaCompletePartialOrder.Chain α
      f : OrderHom α (Part β)
      g : OrderHom α (β → Part γ)
      x : Part γ
      h''' : ∀ (i : Nat), LE.le ((c.map (f.partBind g)) i) x
      b : β
      hb : Membership.mem (OmegaCompletePartialOrder.ωSup (c.map f)) b
      ⊢ LE.le (OmegaCompletePartialOrder.ωSup (c.map g) b) x
    -/
    apply ωSup_le _ _ _
    /-
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      β γ : Type v
      c : OmegaCompletePartialOrder.Chain α
      f : OrderHom α (Part β)
      g : OrderHom α (β → Part γ)
      x : Part γ
      h''' : ∀ (i : Nat), LE.le ((c.map (f.partBind g)) i) x
      b : β
      hb : Membership.mem (OmegaCompletePartialOrder.ωSup (c.map f)) b
      ⊢ ∀ (i : Nat), LE.le (((c.map g).map (Pi.evalOrderHom b)) i) x
    -/
    rintro i y hy
    /-
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      β γ : Type v
      c : OmegaCompletePartialOrder.Chain α
      f : OrderHom α (Part β)
      g : OrderHom α (β → Part γ)
      x : Part γ
      h''' : ∀ (i : Nat), LE.le ((c.map (f.partBind g)) i) x
      b : β
      hb : Membership.mem (OmegaCompletePartialOrder.ωSup (c.map f)) b
      i : Nat
      y : γ
      hy : Membership.mem (((c.map g).map (Pi.evalOrderHom b)) i) y
      ⊢ Membership.mem x y
    -/
    simp only [Part.mem_ωSup] at hb
    /-
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      β γ : Type v
      c : OmegaCompletePartialOrder.Chain α
      f : OrderHom α (Part β)
      g : OrderHom α (β → Part γ)
      x : Part γ
      h''' : ∀ (i : Nat), LE.le ((c.map (f.partBind g)) i) x
      b : β
      i : Nat
      y : γ
      hy : Membership.mem (((c.map g).map (Pi.evalOrderHom b)) i) y
      hb : Membership.mem (c.map f) (Part.some b)
      ⊢ Membership.mem x y
    -/
    rcases hb with ⟨j, hb⟩
    /-
      case intro
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      β γ : Type v
      c : OmegaCompletePartialOrder.Chain α
      f : OrderHom α (Part β)
      g : OrderHom α (β → Part γ)
      x : Part γ
      h''' : ∀ (i : Nat), LE.le ((c.map (f.partBind g)) i) x
      b : β
      i : Nat
      y : γ
      hy : Membership.mem (((c.map g).map (Pi.evalOrderHom b)) i) y
      j : Nat
      hb : Eq (Part.some b) ((c.map f) j)
      ⊢ Membership.mem x y
    -/
    replace hb := hb.symm
    /-
      case intro
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      β γ : Type v
      c : OmegaCompletePartialOrder.Chain α
      f : OrderHom α (Part β)
      g : OrderHom α (β → Part γ)
      x : Part γ
      h''' : ∀ (i : Nat), LE.le ((c.map (f.partBind g)) i) x
      b : β
      i : Nat
      y : γ
      hy : Membership.mem (((c.map g).map (Pi.evalOrderHom b)) i) y
      j : Nat
      hb : Eq ((c.map f) j) (Part.some b)
      ⊢ Membership.mem x y
    -/
    simp only [Part.eq_some_iff, Chain.map_coe, Function.comp_apply, OrderHom.apply_coe] at hy hb
    /-
      case intro
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      β γ : Type v
      c : OmegaCompletePartialOrder.Chain α
      f : OrderHom α (Part β)
      g : OrderHom α (β → Part γ)
      x : Part γ
      h''' : ∀ (i : Nat), LE.le ((c.map (f.partBind g)) i) x
      b : β
      i : Nat
      y : γ
      hy : Membership.mem ((Pi.evalOrderHom b) (g (c i))) y
      j : Nat
      hb : Membership.mem ((c.map f) j) b
      ⊢ Membership.mem x y
    -/
    replace hb : b ∈ f (c (max i j)) := f.mono (c.mono (le_max_right i j)) _ hb
    /-
      case intro
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      β γ : Type v
      c : OmegaCompletePartialOrder.Chain α
      f : OrderHom α (Part β)
      g : OrderHom α (β → Part γ)
      x : Part γ
      h''' : ∀ (i : Nat), LE.le ((c.map (f.partBind g)) i) x
      b : β
      i : Nat
      y : γ
      hy : Membership.mem ((Pi.evalOrderHom b) (g (c i))) y
      j : Nat
      hb : Membership.mem (f (c (Max.max i j))) b
      ⊢ Membership.mem x y
    -/
    replace hy : y ∈ g (c (max i j)) b := g.mono (c.mono (le_max_left i j)) _ _ hy
    /-
      case intro
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      β γ : Type v
      c : OmegaCompletePartialOrder.Chain α
      f : OrderHom α (Part β)
      g : OrderHom α (β → Part γ)
      x : Part γ
      h''' : ∀ (i : Nat), LE.le ((c.map (f.partBind g)) i) x
      b : β
      i : Nat
      y : γ
      j : Nat
      hb : Membership.mem (f (c (Max.max i j))) b
      hy : Membership.mem (g (c (Max.max i j)) b) y
      ⊢ Membership.mem x y
    -/
    apply h''' (max i j)
    simp only [exists_prop, Part.bind_eq_bind, Part.mem_bind_iff, Chain.map_coe,
      Function.comp_apply, OrderHom.partBind_coe]
    /-
      case intro.a
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      β γ : Type v
      c : OmegaCompletePartialOrder.Chain α
      f : OrderHom α (Part β)
      g : OrderHom α (β → Part γ)
      x : Part γ
      h''' : ∀ (i : Nat), LE.le ((c.map (f.partBind g)) i) x
      b : β
      i : Nat
      y : γ
      j : Nat
      hb : Membership.mem (f (c (Max.max i j))) b
      hy : Membership.mem (g (c (Max.max i j)) b) y
      ⊢ Exists fun a => And (Membership.mem (f (c (Max.max i j))) a) (Membership.mem …
    -/
    exact ⟨_, hb, hy⟩
    /-
      🎉 no goals
    -/
    /-
      case H.mpr
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      β γ : Type v
      c : OmegaCompletePartialOrder.Chain α
      f : OrderHom α (Part β)
      g : OrderHom α (β → Part γ)
      x : Part γ
      h''' : ∀ (a : β), Membership.mem (OmegaCompletePartialOrder.ωSup (c.map f)) a  …
      ⊢ ∀ (i : Nat), LE.le ((c.map (f.partBind g)) i) x
    -/
  · intro i
    /-
      case H.mpr
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      β γ : Type v
      c : OmegaCompletePartialOrder.Chain α
      f : OrderHom α (Part β)
      g : OrderHom α (β → Part γ)
      x : Part γ
      h''' : ∀ (a : β), Membership.mem (OmegaCompletePartialOrder.ωSup (c.map f)) a  …
      i : Nat
      ⊢ LE.le ((c.map (f.partBind g)) i) x
    -/
    intro y hy
    simp only [exists_prop, Part.bind_eq_bind, Part.mem_bind_iff, Chain.map_coe,
      Function.comp_apply, OrderHom.partBind_coe] at hy
    /-
      case H.mpr
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      β γ : Type v
      c : OmegaCompletePartialOrder.Chain α
      f : OrderHom α (Part β)
      g : OrderHom α (β → Part γ)
      x : Part γ
      h''' : ∀ (a : β), Membership.mem (OmegaCompletePartialOrder.ωSup (c.map f)) a  …
      i : Nat
      y : γ
      hy : Exists fun a => And (Membership.mem (f (c i)) a) (Membership.mem (g (c i) …
      ⊢ Membership.mem x y
    -/
    rcases hy with ⟨b, hb₀, hb₁⟩
    /-
      case H.mpr.intro.intro
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      β γ : Type v
      c : OmegaCompletePartialOrder.Chain α
      f : OrderHom α (Part β)
      g : OrderHom α (β → Part γ)
      x : Part γ
      h''' : ∀ (a : β), Membership.mem (OmegaCompletePartialOrder.ωSup (c.map f)) a  …
      i : Nat
      y : γ
      b : β
      hb₀ : Membership.mem (f (c i)) b
      hb₁ : Membership.mem (g (c i) b) y
      ⊢ Membership.mem x y
    -/
    apply h''' b _
      /-
        case H.mpr.intro.intro.a
        α : Type u_2
        inst✝ : OmegaCompletePartialOrder α
        β γ : Type v
        c : OmegaCompletePartialOrder.Chain α
        f : OrderHom α (Part β)
        g : OrderHom α (β → Part γ)
        x : Part γ
        h''' : ∀ (a : β), Membership.mem (OmegaCompletePartialOrder.ωSup (c.map f)) a  …
        i : Nat
        y : γ
        b : β
        hb₀ : Membership.mem (f (c i)) b
        hb₁ : Membership.mem (g (c i) b) y
        ⊢ Membership.mem (OmegaCompletePartialOrder.ωSup (c.map g) b) y
      -/
    · apply le_ωSup (c.map g) _ _ _ hb₁
      /-
        🎉 no goals
      -/
      /-
        α : Type u_2
        inst✝ : OmegaCompletePartialOrder α
        β γ : Type v
        c : OmegaCompletePartialOrder.Chain α
        f : OrderHom α (Part β)
        g : OrderHom α (β → Part γ)
        x : Part γ
        h''' : ∀ (a : β), Membership.mem (OmegaCompletePartialOrder.ωSup (c.map f)) a  …
        i : Nat
        y : γ
        b : β
        hb₀ : Membership.mem (f (c i)) b
        hb₁ : Membership.mem (g (c i) b) y
        ⊢ Membership.mem (OmegaCompletePartialOrder.ωSup (c.map f)) b
      -/
    · apply le_ωSup (c.map f) i _ hb₀
      /-
        🎉 no goals
      -/

-- TODO: We should move `ωScottContinuous` to the root namespace

lemma ωScottContinuous.bind {β γ} {f : α → Part β} {g : α → β → Part γ} (hf : ωScottContinuous f)
    (hg : ωScottContinuous g) : ωScottContinuous fun x ↦ f x >>= g x :=
  ωScottContinuous.of_monotone_map_ωSup
                                                  /-
                                                    α : Type u_2
                                                    inst✝ : OmegaCompletePartialOrder α
                                                    β γ : Type u_6
                                                    f : α → Part β
                                                    g : α → β → Part γ
                                                    hf : OmegaCompletePartialOrder.ωScottContinuous f
                                                    hg : OmegaCompletePartialOrder.ωScottContinuous g
                                                    c : OmegaCompletePartialOrder.Chain α
                                                    ⊢ Eq (Bind.bind (f (OmegaCompletePartialOrder.ωSup c)) (g (OmegaCompletePartia …
                                                  -/
    ⟨hf.monotone.partBind hg.monotone, fun c ↦ by rw [hf.map_ωSup, hg.map_ωSup, ← ωSup_bind]; rfl⟩
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/


lemma ωScottContinuous.map {β γ} {f : β → γ} {g : α → Part β} (hg : ωScottContinuous g) :
    ωScottContinuous fun x ↦ f <$> g x := by
  /-
    α : Type u_2
    inst✝ : OmegaCompletePartialOrder α
    β γ : Type u_6
    f : β → γ
    g : α → Part β
    hg : OmegaCompletePartialOrder.ωScottContinuous g
    ⊢ OmegaCompletePartialOrder.ωScottContinuous fun x => Functor.map f (g x)
  -/
  simpa only [map_eq_bind_pure_comp] using ωScottContinuous.bind hg ωScottContinuous.const
  /-
    🎉 no goals
  -/


lemma ωScottContinuous.seq {β γ} {f : α → Part (β → γ)} {g : α → Part β} (hf : ωScottContinuous f)
    (hg : ωScottContinuous g) : ωScottContinuous fun x ↦ f x <*> g x := by
  /-
    α : Type u_2
    inst✝ : OmegaCompletePartialOrder α
    β γ : Type u_6
    f : α → Part (β → γ)
    g : α → Part β
    hf : OmegaCompletePartialOrder.ωScottContinuous f
    hg : OmegaCompletePartialOrder.ωScottContinuous g
    ⊢ OmegaCompletePartialOrder.ωScottContinuous fun x => Seq.seq (f x) fun x_1 => …
  -/
  simp only [seq_eq_bind_map]
  /-
    α : Type u_2
    inst✝ : OmegaCompletePartialOrder α
    β γ : Type u_6
    f : α → Part (β → γ)
    g : α → Part β
    hf : OmegaCompletePartialOrder.ωScottContinuous f
    hg : OmegaCompletePartialOrder.ωScottContinuous g
    ⊢ OmegaCompletePartialOrder.ωScottContinuous fun x => Bind.bind (f x) fun x_1  …
  -/
  exact ωScottContinuous.bind hf <| ωScottContinuous.of_apply₂ fun _ ↦ ωScottContinuous.map hg
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated ωScottContinuous.bind (since := "2024-05-29")]
theorem bind_continuous' {β γ : Type v} (f : α → Part β) (g : α → β → Part γ) :
    Continuous' f → Continuous' g → Continuous' fun x => f x >>= g x
  | ⟨hf, hf'⟩, ⟨hg, hg'⟩ =>
    Continuous.of_bundled' (OrderHom.partBind ⟨f, hf⟩ ⟨g, hg⟩)
          /-
            α : Type u_2
            inst✝ : OmegaCompletePartialOrder α
            β γ : Type v
            f : α → Part β
            g : α → β → Part γ
            hf : Monotone f
            hf' : OmegaCompletePartialOrder.Continuous { toFun := f, monotone' := hf }
            hg : Monotone g
            hg' : OmegaCompletePartialOrder.Continuous { toFun := g, monotone' := hg }
            ⊢ OmegaCompletePartialOrder.Continuous ({ toFun := f, monotone' := hf }.partBi …
          -/
      (by intro c; rw [ωSup_bind, ← hf', ← hg']; rfl)
                                                 /-
                                                   🎉 no goals
                                                 -/


set_option linter.deprecated false in
@[deprecated ωScottContinuous.map (since := "2024-05-29")]
theorem map_continuous' {β γ : Type v} (f : β → γ) (g : α → Part β) (hg : Continuous' g) :
    Continuous' fun x => f <$> g x := by
  /-
    α : Type u_2
    inst✝ : OmegaCompletePartialOrder α
    β γ : Type v
    f : β → γ
    g : α → Part β
    hg : OmegaCompletePartialOrder.Continuous' g
    ⊢ OmegaCompletePartialOrder.Continuous' fun x => Functor.map f (g x)
  -/
  simp only [map_eq_bind_pure_comp]; apply bind_continuous' _ _ hg; apply const_continuous'
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


set_option linter.deprecated false in
@[deprecated ωScottContinuous.seq (since := "2024-05-29")]
theorem seq_continuous' {β γ : Type v} (f : α → Part (β → γ)) (g : α → Part β) (hf : Continuous' f)
    (hg : Continuous' g) : Continuous' fun x => f x <*> g x := by
  /-
    α : Type u_2
    inst✝ : OmegaCompletePartialOrder α
    β γ : Type v
    f : α → Part (β → γ)
    g : α → Part β
    hf : OmegaCompletePartialOrder.Continuous' f
    hg : OmegaCompletePartialOrder.Continuous' g
    ⊢ OmegaCompletePartialOrder.Continuous' fun x => Seq.seq (f x) fun x_1 => g x
  -/
  simp only [seq_eq_bind_map]
  /-
    α : Type u_2
    inst✝ : OmegaCompletePartialOrder α
    β γ : Type v
    f : α → Part (β → γ)
    g : α → Part β
    hf : OmegaCompletePartialOrder.Continuous' f
    hg : OmegaCompletePartialOrder.Continuous' g
    ⊢ OmegaCompletePartialOrder.Continuous' fun x => Bind.bind (f x) fun x_1 => Fu …
  -/
  apply bind_continuous' _ _ hf
  /-
    α : Type u_2
    inst✝ : OmegaCompletePartialOrder α
    β γ : Type v
    f : α → Part (β → γ)
    g : α → Part β
    hf : OmegaCompletePartialOrder.Continuous' f
    hg : OmegaCompletePartialOrder.Continuous' g
    ⊢ OmegaCompletePartialOrder.Continuous' fun x x_1 => Functor.map x_1 (g x)
  -/
  apply OmegaCompletePartialOrder.flip₂_continuous'
  /-
    case hf
    α : Type u_2
    inst✝ : OmegaCompletePartialOrder α
    β γ : Type v
    f : α → Part (β → γ)
    g : α → Part β
    hf : OmegaCompletePartialOrder.Continuous' f
    hg : OmegaCompletePartialOrder.Continuous' g
    ⊢ ∀ (x : β → γ), OmegaCompletePartialOrder.Continuous' fun g_1 => Functor.map  …
  -/
  intro
  /-
    case hf
    α : Type u_2
    inst✝ : OmegaCompletePartialOrder α
    β γ : Type v
    f : α → Part (β → γ)
    g : α → Part β
    hf : OmegaCompletePartialOrder.Continuous' f
    hg : OmegaCompletePartialOrder.Continuous' g
    x✝ : β → γ
    ⊢ OmegaCompletePartialOrder.Continuous' fun g_1 => Functor.map x✝ (g g_1)
  -/
  apply map_continuous' _ _ hg
  /-
    🎉 no goals
  -/


theorem continuous (F : α →𝒄 β) (C : Chain α) : F (ωSup C) = ωSup (C.map F) :=
  F.ωScottContinuous.map_ωSup _


/-- Construct a continuous function from a bare function, a continuous function, and a proof that
they are equal. -/
-- Porting note: removed `@[reducible]`
@[simps!]
def copy (f : α → β) (g : α →𝒄 β) (h : f = g) : α →𝒄 β where
  toOrderHom := g.1.copy f h
                  /-
                    ι : Sort u_1
                    α : Type u_2
                    β : Type u_3
                    γ : Type u_4
                    δ : Type u_5
                    inst✝³ : OmegaCompletePartialOrder α
                    inst✝² : OmegaCompletePartialOrder β
                    inst✝¹ : OmegaCompletePartialOrder γ
                    inst✝ : OmegaCompletePartialOrder δ
                    f : α → β
                    g : OmegaCompletePartialOrder.ContinuousHom α β
                    h : Eq f ⇑g
                    ⊢ ∀ (c : OmegaCompletePartialOrder.Chain α), Eq ((g.copy f h).toFun (OmegaComp …
                  -/
  map_ωSup' := by rw [OrderHom.copy_eq]; exact g.map_ωSup'
                                         /-
                                           🎉 no goals
                                         -/

-- Porting note: `of_mono` now defeq `mk`


/-- The identity as a continuous function. -/
@[simps!]
def id : α →𝒄 α := ⟨OrderHom.id, ωScottContinuous.id.map_ωSup⟩


/-- The composition of continuous functions. -/
@[simps!]
def comp (f : β →𝒄 γ) (g : α →𝒄 β) : α →𝒄 γ :=
  ⟨.comp f.1 g.1, (f.ωScottContinuous.comp g.ωScottContinuous).map_ωSup⟩


@[ext]
protected theorem ext (f g : α →𝒄 β) (h : ∀ x, f x = g x) : f = g := DFunLike.ext f g h


protected theorem coe_inj (f g : α →𝒄 β) (h : (f : α → β) = g) : f = g :=
  DFunLike.ext' h


@[simp]
theorem comp_id (f : β →𝒄 γ) : f.comp id = f := rfl


@[simp]
theorem id_comp (f : β →𝒄 γ) : id.comp f = f := rfl


@[simp]
theorem comp_assoc (f : γ →𝒄 δ) (g : β →𝒄 γ) (h : α →𝒄 β) : f.comp (g.comp h) = (f.comp g).comp h :=
  rfl


@[simp]
theorem coe_apply (a : α) (f : α →𝒄 β) : (f : α →o β) a = f a :=
  rfl


/-- `Function.const` is a continuous function. -/
@[simps!]
def const (x : β) : α →𝒄 β := ⟨.const _ x, ωScottContinuous.const.map_ωSup⟩


instance [Inhabited β] : Inhabited (α →𝒄 β) :=
  ⟨const default⟩


/-- The map from continuous functions to monotone functions is itself a monotone function. -/
@[simps]
def toMono : (α →𝒄 β) →o α →o β where
  toFun f := f
  monotone' _ _ h := h


/-- When proving that a chain of applications is below a bound `z`, it suffices to consider the
functions and values being selected from the same index in the chains.

This lemma is more specific than necessary, i.e. `c₀` only needs to be a
chain of monotone functions, but it is only used with continuous functions. -/
@[simp]
theorem forall_forall_merge (c₀ : Chain (α →𝒄 β)) (c₁ : Chain α) (z : β) :
    (∀ i j : ℕ, (c₀ i) (c₁ j) ≤ z) ↔ ∀ i : ℕ, (c₀ i) (c₁ i) ≤ z := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : OmegaCompletePartialOrder β
    c₀ : OmegaCompletePartialOrder.Chain (OmegaCompletePartialOrder.ContinuousHom  …
    c₁ : OmegaCompletePartialOrder.Chain α
    z : β
    ⊢ Iff (∀ (i j : Nat), LE.le ((c₀ i) (c₁ j)) z) (∀ (i : Nat), LE.le ((c₀ i) (c₁ …
  -/
  constructor <;> introv h
    /-
      case mp
      α : Type u_2
      β : Type u_3
      inst✝¹ : OmegaCompletePartialOrder α
      inst✝ : OmegaCompletePartialOrder β
      c₀ : OmegaCompletePartialOrder.Chain (OmegaCompletePartialOrder.ContinuousHom  …
      c₁ : OmegaCompletePartialOrder.Chain α
      z : β
      h : ∀ (i j : Nat), LE.le ((c₀ i) (c₁ j)) z
      i : Nat
      ⊢ LE.le ((c₀ i) (c₁ i)) z
    -/
  · apply h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_2
      β : Type u_3
      inst✝¹ : OmegaCompletePartialOrder α
      inst✝ : OmegaCompletePartialOrder β
      c₀ : OmegaCompletePartialOrder.Chain (OmegaCompletePartialOrder.ContinuousHom  …
      c₁ : OmegaCompletePartialOrder.Chain α
      z : β
      h : ∀ (i : Nat), LE.le ((c₀ i) (c₁ i)) z
      i j : Nat
      ⊢ LE.le ((c₀ i) (c₁ j)) z
    -/
  · apply le_trans _ (h (max i j))
    /-
      α : Type u_2
      β : Type u_3
      inst✝¹ : OmegaCompletePartialOrder α
      inst✝ : OmegaCompletePartialOrder β
      c₀ : OmegaCompletePartialOrder.Chain (OmegaCompletePartialOrder.ContinuousHom  …
      c₁ : OmegaCompletePartialOrder.Chain α
      z : β
      h : ∀ (i : Nat), LE.le ((c₀ i) (c₁ i)) z
      i j : Nat
      ⊢ LE.le ((c₀ i) (c₁ j)) ((c₀ (Max.max i j)) (c₁ (Max.max i j)))
    -/
    trans c₀ i (c₁ (max i j))
      /-
        α : Type u_2
        β : Type u_3
        inst✝¹ : OmegaCompletePartialOrder α
        inst✝ : OmegaCompletePartialOrder β
        c₀ : OmegaCompletePartialOrder.Chain (OmegaCompletePartialOrder.ContinuousHom  …
        c₁ : OmegaCompletePartialOrder.Chain α
        z : β
        h : ∀ (i : Nat), LE.le ((c₀ i) (c₁ i)) z
        i j : Nat
        ⊢ LE.le ((c₀ i) (c₁ j)) ((c₀ i) (c₁ (Max.max i j)))
      -/
    · apply (c₀ i).monotone
      /-
        case a
        α : Type u_2
        β : Type u_3
        inst✝¹ : OmegaCompletePartialOrder α
        inst✝ : OmegaCompletePartialOrder β
        c₀ : OmegaCompletePartialOrder.Chain (OmegaCompletePartialOrder.ContinuousHom  …
        c₁ : OmegaCompletePartialOrder.Chain α
        z : β
        h : ∀ (i : Nat), LE.le ((c₀ i) (c₁ i)) z
        i j : Nat
        ⊢ LE.le (c₁ j) (c₁ (Max.max i j))
      -/
      apply c₁.monotone
      /-
        case a.a
        α : Type u_2
        β : Type u_3
        inst✝¹ : OmegaCompletePartialOrder α
        inst✝ : OmegaCompletePartialOrder β
        c₀ : OmegaCompletePartialOrder.Chain (OmegaCompletePartialOrder.ContinuousHom  …
        c₁ : OmegaCompletePartialOrder.Chain α
        z : β
        h : ∀ (i : Nat), LE.le ((c₀ i) (c₁ i)) z
        i j : Nat
        ⊢ LE.le j (Max.max i j)
      -/
      apply le_max_right
      /-
        🎉 no goals
      -/
      /-
        α : Type u_2
        β : Type u_3
        inst✝¹ : OmegaCompletePartialOrder α
        inst✝ : OmegaCompletePartialOrder β
        c₀ : OmegaCompletePartialOrder.Chain (OmegaCompletePartialOrder.ContinuousHom  …
        c₁ : OmegaCompletePartialOrder.Chain α
        z : β
        h : ∀ (i : Nat), LE.le ((c₀ i) (c₁ i)) z
        i j : Nat
        ⊢ LE.le ((c₀ i) (c₁ (Max.max i j))) ((c₀ (Max.max i j)) (c₁ (Max.max i j)))
      -/
    · apply c₀.monotone
      /-
        case a
        α : Type u_2
        β : Type u_3
        inst✝¹ : OmegaCompletePartialOrder α
        inst✝ : OmegaCompletePartialOrder β
        c₀ : OmegaCompletePartialOrder.Chain (OmegaCompletePartialOrder.ContinuousHom  …
        c₁ : OmegaCompletePartialOrder.Chain α
        z : β
        h : ∀ (i : Nat), LE.le ((c₀ i) (c₁ i)) z
        i j : Nat
        ⊢ LE.le i (Max.max i j)
      -/
      apply le_max_left
      /-
        🎉 no goals
      -/


@[simp]
theorem forall_forall_merge' (c₀ : Chain (α →𝒄 β)) (c₁ : Chain α) (z : β) :
    (∀ j i : ℕ, (c₀ i) (c₁ j) ≤ z) ↔ ∀ i : ℕ, (c₀ i) (c₁ i) ≤ z := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : OmegaCompletePartialOrder α
    inst✝ : OmegaCompletePartialOrder β
    c₀ : OmegaCompletePartialOrder.Chain (OmegaCompletePartialOrder.ContinuousHom  …
    c₁ : OmegaCompletePartialOrder.Chain α
    z : β
    ⊢ Iff (∀ (j i : Nat), LE.le ((c₀ i) (c₁ j)) z) (∀ (i : Nat), LE.le ((c₀ i) (c₁ …
  -/
  rw [forall_swap, forall_forall_merge]
  /-
    🎉 no goals
  -/


/-- The `ωSup` operator for continuous functions, which takes the pointwise countable supremum
of the functions in the `ω`-chain. -/
@[simps!]
protected def ωSup (c : Chain (α →𝒄 β)) : α →𝒄 β where
  toOrderHom := ωSup <| c.map toMono
                                                 /-
                                                   ι : Sort u_1
                                                   α : Type u_2
                                                   β : Type u_3
                                                   γ : Type u_4
                                                   δ : Type u_5
                                                   inst✝³ : OmegaCompletePartialOrder α
                                                   inst✝² : OmegaCompletePartialOrder β
                                                   inst✝¹ : OmegaCompletePartialOrder γ
                                                   inst✝ : OmegaCompletePartialOrder δ
                                                   c : OmegaCompletePartialOrder.Chain (OmegaCompletePartialOrder.ContinuousHom α …
                                                   c' : OmegaCompletePartialOrder.Chain α
                                                   a : β
                                                   ⊢ Iff (LE.le ((OmegaCompletePartialOrder.ωSup (c.map OmegaCompletePartialOrder …
                                                 -/
  map_ωSup' c' := eq_of_forall_ge_iff fun a ↦ by simp [(c _).ωScottContinuous.map_ωSup]
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simps ωSup]
instance : OmegaCompletePartialOrder (α →𝒄 β) :=
  OmegaCompletePartialOrder.lift ContinuousHom.toMono ContinuousHom.ωSup
    (fun _ _ h => h) (fun _ => rfl)


/-- The application of continuous functions as a continuous function. -/
@[simps]
def apply : (α →𝒄 β) × α →𝒄 β where
  toFun f := f.1 f.2
  monotone' x y h := by
    /-
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝³ : OmegaCompletePartialOrder α
      inst✝² : OmegaCompletePartialOrder β
      inst✝¹ : OmegaCompletePartialOrder γ
      inst✝ : OmegaCompletePartialOrder δ
      x y : Prod (OmegaCompletePartialOrder.ContinuousHom α β) α
      h : LE.le x y
      ⊢ LE.le ((fun f => f.1 f.2) x) ((fun f => f.1 f.2) y)
    -/
    dsimp
    /-
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝³ : OmegaCompletePartialOrder α
      inst✝² : OmegaCompletePartialOrder β
      inst✝¹ : OmegaCompletePartialOrder γ
      inst✝ : OmegaCompletePartialOrder δ
      x y : Prod (OmegaCompletePartialOrder.ContinuousHom α β) α
      h : LE.le x y
      ⊢ LE.le (x.1 x.2) (y.1 y.2)
    -/
    trans y.fst x.snd <;> [apply h.1; apply y.1.monotone h.2]
    /-
      🎉 no goals
    -/
  map_ωSup' c := by
    /-
      ι : Sort u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      δ : Type u_5
      inst✝³ : OmegaCompletePartialOrder α
      inst✝² : OmegaCompletePartialOrder β
      inst✝¹ : OmegaCompletePartialOrder γ
      inst✝ : OmegaCompletePartialOrder δ
      c : OmegaCompletePartialOrder.Chain (Prod (OmegaCompletePartialOrder.Continuou …
      ⊢ Eq ({ toFun := fun f => f.1 f.2, monotone' := ⋯ }.toFun (OmegaCompletePartia …
    -/
    apply le_antisymm
      /-
        case a
        ι : Sort u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        δ : Type u_5
        inst✝³ : OmegaCompletePartialOrder α
        inst✝² : OmegaCompletePartialOrder β
        inst✝¹ : OmegaCompletePartialOrder γ
        inst✝ : OmegaCompletePartialOrder δ
        c : OmegaCompletePartialOrder.Chain (Prod (OmegaCompletePartialOrder.Continuou …
        ⊢ LE.le ({ toFun := fun f => f.1 f.2, monotone' := ⋯ }.toFun (OmegaCompletePar …
      -/
    · apply ωSup_le
      /-
        case a.a
        ι : Sort u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        δ : Type u_5
        inst✝³ : OmegaCompletePartialOrder α
        inst✝² : OmegaCompletePartialOrder β
        inst✝¹ : OmegaCompletePartialOrder γ
        inst✝ : OmegaCompletePartialOrder δ
        c : OmegaCompletePartialOrder.Chain (Prod (OmegaCompletePartialOrder.Continuou …
        ⊢ ∀ (i : Nat), LE.le ((((c.map OrderHom.fst).map OmegaCompletePartialOrder.Con …
      -/
      intro i
      /-
        case a.a
        ι : Sort u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        δ : Type u_5
        inst✝³ : OmegaCompletePartialOrder α
        inst✝² : OmegaCompletePartialOrder β
        inst✝¹ : OmegaCompletePartialOrder γ
        inst✝ : OmegaCompletePartialOrder δ
        c : OmegaCompletePartialOrder.Chain (Prod (OmegaCompletePartialOrder.Continuou …
        i : Nat
        ⊢ LE.le ((((c.map OrderHom.fst).map OmegaCompletePartialOrder.ContinuousHom.to …
      -/
      dsimp
      /-
        case a.a
        ι : Sort u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        δ : Type u_5
        inst✝³ : OmegaCompletePartialOrder α
        inst✝² : OmegaCompletePartialOrder β
        inst✝¹ : OmegaCompletePartialOrder γ
        inst✝ : OmegaCompletePartialOrder δ
        c : OmegaCompletePartialOrder.Chain (Prod (OmegaCompletePartialOrder.Continuou …
        i : Nat
        ⊢ LE.le ((c i).1 (OmegaCompletePartialOrder.ωSup (c.map OrderHom.snd))) (Omega …
      -/
      rw [(c _).fst.continuous]
      /-
        case a.a
        ι : Sort u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        δ : Type u_5
        inst✝³ : OmegaCompletePartialOrder α
        inst✝² : OmegaCompletePartialOrder β
        inst✝¹ : OmegaCompletePartialOrder γ
        inst✝ : OmegaCompletePartialOrder δ
        c : OmegaCompletePartialOrder.Chain (Prod (OmegaCompletePartialOrder.Continuou …
        i : Nat
        ⊢ LE.le (OmegaCompletePartialOrder.ωSup ((c.map OrderHom.snd).map ↑(c i).1)) ( …
      -/
      apply ωSup_le
      /-
        case a.a.a
        ι : Sort u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        δ : Type u_5
        inst✝³ : OmegaCompletePartialOrder α
        inst✝² : OmegaCompletePartialOrder β
        inst✝¹ : OmegaCompletePartialOrder γ
        inst✝ : OmegaCompletePartialOrder δ
        c : OmegaCompletePartialOrder.Chain (Prod (OmegaCompletePartialOrder.Continuou …
        i : Nat
        ⊢ ∀ (i_1 : Nat), LE.le (((c.map OrderHom.snd).map ↑(c i).1) i_1) (OmegaComplet …
      -/
      intro j
      /-
        case a.a.a
        ι : Sort u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        δ : Type u_5
        inst✝³ : OmegaCompletePartialOrder α
        inst✝² : OmegaCompletePartialOrder β
        inst✝¹ : OmegaCompletePartialOrder γ
        inst✝ : OmegaCompletePartialOrder δ
        c : OmegaCompletePartialOrder.Chain (Prod (OmegaCompletePartialOrder.Continuou …
        i j : Nat
        ⊢ LE.le (((c.map OrderHom.snd).map ↑(c i).1) j) (OmegaCompletePartialOrder.ωSu …
      -/
      apply le_ωSup_of_le (max i j)
      /-
        case a.a.a
        ι : Sort u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        δ : Type u_5
        inst✝³ : OmegaCompletePartialOrder α
        inst✝² : OmegaCompletePartialOrder β
        inst✝¹ : OmegaCompletePartialOrder γ
        inst✝ : OmegaCompletePartialOrder δ
        c : OmegaCompletePartialOrder.Chain (Prod (OmegaCompletePartialOrder.Continuou …
        i j : Nat
        ⊢ LE.le (((c.map OrderHom.snd).map ↑(c i).1) j) ((c.map { toFun := fun f => f. …
      -/
      apply apply_mono
        /-
          case a.a.a.h₁
          ι : Sort u_1
          α : Type u_2
          β : Type u_3
          γ : Type u_4
          δ : Type u_5
          inst✝³ : OmegaCompletePartialOrder α
          inst✝² : OmegaCompletePartialOrder β
          inst✝¹ : OmegaCompletePartialOrder γ
          inst✝ : OmegaCompletePartialOrder δ
          c : OmegaCompletePartialOrder.Chain (Prod (OmegaCompletePartialOrder.Continuou …
          i j : Nat
          ⊢ LE.le (c i).1 (c (Max.max i j)).1
        -/
      · exact monotone_fst (OrderHom.mono _ (le_max_left _ _))
        /-
          🎉 no goals
        -/
        /-
          case a.a.a.h₂
          ι : Sort u_1
          α : Type u_2
          β : Type u_3
          γ : Type u_4
          δ : Type u_5
          inst✝³ : OmegaCompletePartialOrder α
          inst✝² : OmegaCompletePartialOrder β
          inst✝¹ : OmegaCompletePartialOrder γ
          inst✝ : OmegaCompletePartialOrder δ
          c : OmegaCompletePartialOrder.Chain (Prod (OmegaCompletePartialOrder.Continuou …
          i j : Nat
          ⊢ LE.le ((c.map OrderHom.snd) j) (c (Max.max i j)).2
        -/
      · exact monotone_snd (OrderHom.mono _ (le_max_right _ _))
        /-
          🎉 no goals
        -/
      /-
        case a
        ι : Sort u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        δ : Type u_5
        inst✝³ : OmegaCompletePartialOrder α
        inst✝² : OmegaCompletePartialOrder β
        inst✝¹ : OmegaCompletePartialOrder γ
        inst✝ : OmegaCompletePartialOrder δ
        c : OmegaCompletePartialOrder.Chain (Prod (OmegaCompletePartialOrder.Continuou …
        ⊢ LE.le (OmegaCompletePartialOrder.ωSup (c.map { toFun := fun f => f.1 f.2, mo …
      -/
    · apply ωSup_le
      /-
        case a.a
        ι : Sort u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        δ : Type u_5
        inst✝³ : OmegaCompletePartialOrder α
        inst✝² : OmegaCompletePartialOrder β
        inst✝¹ : OmegaCompletePartialOrder γ
        inst✝ : OmegaCompletePartialOrder δ
        c : OmegaCompletePartialOrder.Chain (Prod (OmegaCompletePartialOrder.Continuou …
        ⊢ ∀ (i : Nat), LE.le ((c.map { toFun := fun f => f.1 f.2, monotone' := ⋯ }) i) …
      -/
      intro i
      /-
        case a.a
        ι : Sort u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        δ : Type u_5
        inst✝³ : OmegaCompletePartialOrder α
        inst✝² : OmegaCompletePartialOrder β
        inst✝¹ : OmegaCompletePartialOrder γ
        inst✝ : OmegaCompletePartialOrder δ
        c : OmegaCompletePartialOrder.Chain (Prod (OmegaCompletePartialOrder.Continuou …
        i : Nat
        ⊢ LE.le ((c.map { toFun := fun f => f.1 f.2, monotone' := ⋯ }) i) ({ toFun :=  …
      -/
      apply le_ωSup_of_le i
      /-
        case a.a
        ι : Sort u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        δ : Type u_5
        inst✝³ : OmegaCompletePartialOrder α
        inst✝² : OmegaCompletePartialOrder β
        inst✝¹ : OmegaCompletePartialOrder γ
        inst✝ : OmegaCompletePartialOrder δ
        c : OmegaCompletePartialOrder.Chain (Prod (OmegaCompletePartialOrder.Continuou …
        i : Nat
        ⊢ LE.le ((c.map { toFun := fun f => f.1 f.2, monotone' := ⋯ }) i) ((((c.map Or …
      -/
      dsimp
      /-
        case a.a
        ι : Sort u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        δ : Type u_5
        inst✝³ : OmegaCompletePartialOrder α
        inst✝² : OmegaCompletePartialOrder β
        inst✝¹ : OmegaCompletePartialOrder γ
        inst✝ : OmegaCompletePartialOrder δ
        c : OmegaCompletePartialOrder.Chain (Prod (OmegaCompletePartialOrder.Continuou …
        i : Nat
        ⊢ LE.le ((c i).1 (c i).2) ((c i).1 (OmegaCompletePartialOrder.ωSup (c.map Orde …
      -/
      apply OrderHom.mono _
      /-
        case a.a.a
        ι : Sort u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        δ : Type u_5
        inst✝³ : OmegaCompletePartialOrder α
        inst✝² : OmegaCompletePartialOrder β
        inst✝¹ : OmegaCompletePartialOrder γ
        inst✝ : OmegaCompletePartialOrder δ
        c : OmegaCompletePartialOrder.Chain (Prod (OmegaCompletePartialOrder.Continuou …
        i : Nat
        ⊢ LE.le (c i).2 (OmegaCompletePartialOrder.ωSup (c.map OrderHom.snd))
      -/
      apply le_ωSup_of_le i
      /-
        case a.a.a
        ι : Sort u_1
        α : Type u_2
        β : Type u_3
        γ : Type u_4
        δ : Type u_5
        inst✝³ : OmegaCompletePartialOrder α
        inst✝² : OmegaCompletePartialOrder β
        inst✝¹ : OmegaCompletePartialOrder γ
        inst✝ : OmegaCompletePartialOrder δ
        c : OmegaCompletePartialOrder.Chain (Prod (OmegaCompletePartialOrder.Continuou …
        i : Nat
        ⊢ LE.le (c i).2 ((c.map OrderHom.snd) i)
      -/
      rfl
      /-
        🎉 no goals
      -/


theorem ωSup_def (c : Chain (α →𝒄 β)) (x : α) : ωSup c x = ContinuousHom.ωSup c x :=
  rfl


theorem ωSup_apply_ωSup (c₀ : Chain (α →𝒄 β)) (c₁ : Chain α) :
                                                            /-
                                                              α : Type u_2
                                                              β : Type u_3
                                                              inst✝¹ : OmegaCompletePartialOrder α
                                                              inst✝ : OmegaCompletePartialOrder β
                                                              c₀ : OmegaCompletePartialOrder.Chain (OmegaCompletePartialOrder.ContinuousHom  …
                                                              c₁ : OmegaCompletePartialOrder.Chain α
                                                              ⊢ Eq ((OmegaCompletePartialOrder.ωSup c₀) (OmegaCompletePartialOrder.ωSup c₁)) …
                                                            -/
    ωSup c₀ (ωSup c₁) = Prod.apply (ωSup (c₀.zip c₁)) := by simp [Prod.apply_apply, Prod.ωSup_zip]
                                                            /-
                                                              🎉 no goals
                                                            -/


/-- A family of continuous functions yields a continuous family of functions. -/
@[simps]
def flip {α : Type*} (f : α → β →𝒄 γ) : β →𝒄 α → γ where
  toFun x y := f y x
  monotone' _ _ h a := (f a).monotone h
                    /-
                      ι : Sort u_1
                      α✝ : Type u_2
                      β : Type u_3
                      γ : Type u_4
                      δ : Type u_5
                      inst✝³ : OmegaCompletePartialOrder α✝
                      inst✝² : OmegaCompletePartialOrder β
                      inst✝¹ : OmegaCompletePartialOrder γ
                      inst✝ : OmegaCompletePartialOrder δ
                      α : Type u_6
                      f : α → OmegaCompletePartialOrder.ContinuousHom β γ
                      x✝ : OmegaCompletePartialOrder.Chain β
                      ⊢ Eq ({ toFun := fun x y => (f y) x, monotone' := ⋯ }.toFun (OmegaCompletePart …
                    -/
  map_ωSup' _ := by ext x; change f _ _ = _; rw [(f _).continuous]; rfl
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- `Part.bind` as a continuous function. -/
@[simps! apply] -- Porting note: removed `(config := { rhsMd := reducible })`
noncomputable def bind {β γ : Type v} (f : α →𝒄 Part β) (g : α →𝒄 β → Part γ) : α →𝒄 Part γ :=
  .mk (OrderHom.partBind f g.toOrderHom) fun c => by
    /-
      ι : Sort u_1
      α : Type u_2
      β✝ : Type u_3
      γ✝ : Type u_4
      δ : Type u_5
      inst✝³ : OmegaCompletePartialOrder α
      inst✝² : OmegaCompletePartialOrder β✝
      inst✝¹ : OmegaCompletePartialOrder γ✝
      inst✝ : OmegaCompletePartialOrder δ
      β γ : Type v
      f : OmegaCompletePartialOrder.ContinuousHom α (Part β)
      g : OmegaCompletePartialOrder.ContinuousHom α (β → Part γ)
      c : OmegaCompletePartialOrder.Chain α
      ⊢ Eq (((↑f).partBind g.toOrderHom).toFun (OmegaCompletePartialOrder.ωSup c)) ( …
    -/
    rw [ωSup_bind, ← f.continuous, g.toOrderHom_eq_coe, ← g.continuous]
    /-
      ι : Sort u_1
      α : Type u_2
      β✝ : Type u_3
      γ✝ : Type u_4
      δ : Type u_5
      inst✝³ : OmegaCompletePartialOrder α
      inst✝² : OmegaCompletePartialOrder β✝
      inst✝¹ : OmegaCompletePartialOrder γ✝
      inst✝ : OmegaCompletePartialOrder δ
      β γ : Type v
      f : OmegaCompletePartialOrder.ContinuousHom α (Part β)
      g : OmegaCompletePartialOrder.ContinuousHom α (β → Part γ)
      c : OmegaCompletePartialOrder.Chain α
      ⊢ Eq (((↑f).partBind ↑g).toFun (OmegaCompletePartialOrder.ωSup c)) (Bind.bind  …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- `Part.map` as a continuous function. -/
@[simps! apply] -- Porting note: removed `(config := { rhsMd := reducible })`
noncomputable def map {β γ : Type v} (f : β → γ) (g : α →𝒄 Part β) : α →𝒄 Part γ :=
  .copy (fun x => f <$> g x) (bind g (const (pure ∘ f))) <| by
    /-
      ι : Sort u_1
      α : Type u_2
      β✝ : Type u_3
      γ✝ : Type u_4
      δ : Type u_5
      inst✝³ : OmegaCompletePartialOrder α
      inst✝² : OmegaCompletePartialOrder β✝
      inst✝¹ : OmegaCompletePartialOrder γ✝
      inst✝ : OmegaCompletePartialOrder δ
      β γ : Type v
      f : β → γ
      g : OmegaCompletePartialOrder.ContinuousHom α (Part β)
      ⊢ Eq (fun x => Functor.map f (g x)) ⇑(g.bind (OmegaCompletePartialOrder.Contin …
    -/
    ext1
    simp only [map_eq_bind_pure_comp, bind, coe_mk, OrderHom.partBind_coe, coe_apply,
      coe_toOrderHom, const_apply, Part.bind_eq_bind]


/-- `Part.seq` as a continuous function. -/
@[simps! apply] -- Porting note: removed `(config := { rhsMd := reducible })`
noncomputable def seq {β γ : Type v} (f : α →𝒄 Part (β → γ)) (g : α →𝒄 Part β) : α →𝒄 Part γ :=
  .copy (fun x => f x <*> g x) (bind f <| flip <| _root_.flip map g) <| by
      /-
        ι : Sort u_1
        α : Type u_2
        β✝ : Type u_3
        γ✝ : Type u_4
        δ : Type u_5
        inst✝³ : OmegaCompletePartialOrder α
        inst✝² : OmegaCompletePartialOrder β✝
        inst✝¹ : OmegaCompletePartialOrder γ✝
        inst✝ : OmegaCompletePartialOrder δ
        β γ : Type v
        f : OmegaCompletePartialOrder.ContinuousHom α (Part (β → γ))
        g : OmegaCompletePartialOrder.ContinuousHom α (Part β)
        ⊢ Eq (fun x => Seq.seq (f x) fun x_1 => g x) ⇑(f.bind (OmegaCompletePartialOrd …
      -/
      ext
      simp only [seq_eq_bind_map, Part.bind_eq_bind, Part.mem_bind_iff, flip_apply, _root_.flip,
        map_apply, bind_apply, Part.map_eq_map]


/-- Iteration of a function on an initial element interpreted as a chain. -/
def iterateChain (f : α →o α) (x : α) (h : x ≤ f x) : Chain α :=
  ⟨fun n => f^[n] x, f.monotone.monotone_iterate_of_le_map h⟩


/-- The supremum of iterating a function on x arbitrary often is a fixed point -/
theorem ωSup_iterate_mem_fixedPoint (h : x ≤ f x) :
    ωSup (iterateChain f x h) ∈ fixedPoints f := by
  /-
    α : Type u_2
    inst✝ : OmegaCompletePartialOrder α
    f : OmegaCompletePartialOrder.ContinuousHom α α
    x : α
    h : LE.le x (f x)
    ⊢ Membership.mem (Function.fixedPoints ⇑f) (OmegaCompletePartialOrder.ωSup (Om …
  -/
  rw [mem_fixedPoints, IsFixedPt, f.continuous]
  /-
    α : Type u_2
    inst✝ : OmegaCompletePartialOrder α
    f : OmegaCompletePartialOrder.ContinuousHom α α
    x : α
    h : LE.le x (f x)
    ⊢ Eq (OmegaCompletePartialOrder.ωSup ((OmegaCompletePartialOrder.fixedPoints.i …
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      f : OmegaCompletePartialOrder.ContinuousHom α α
      x : α
      h : LE.le x (f x)
      ⊢ LE.le (OmegaCompletePartialOrder.ωSup ((OmegaCompletePartialOrder.fixedPoint …
    -/
  · apply ωSup_le
    /-
      case a.a
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      f : OmegaCompletePartialOrder.ContinuousHom α α
      x : α
      h : LE.le x (f x)
      ⊢ ∀ (i : Nat), LE.le (((OmegaCompletePartialOrder.fixedPoints.iterateChain (↑f …
    -/
    intro n
    /-
      case a.a
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      f : OmegaCompletePartialOrder.ContinuousHom α α
      x : α
      h : LE.le x (f x)
      n : Nat
      ⊢ LE.le (((OmegaCompletePartialOrder.fixedPoints.iterateChain (↑f) x h).map ↑f …
    -/
    simp only [Chain.map_coe, OrderHomClass.coe_coe, comp_apply]
    have : iterateChain f x h (n.succ) = f (iterateChain f x h n) :=
      Function.iterate_succ_apply' ..
    /-
      case a.a
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      f : OmegaCompletePartialOrder.ContinuousHom α α
      x : α
      h : LE.le x (f x)
      n : Nat
      this : Eq ((OmegaCompletePartialOrder.fixedPoints.iterateChain (↑f) x h) n.suc …
      ⊢ LE.le (f ((OmegaCompletePartialOrder.fixedPoints.iterateChain (↑f) x h) n))  …
    -/
    rw [← this]
    /-
      case a.a
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      f : OmegaCompletePartialOrder.ContinuousHom α α
      x : α
      h : LE.le x (f x)
      n : Nat
      this : Eq ((OmegaCompletePartialOrder.fixedPoints.iterateChain (↑f) x h) n.suc …
      ⊢ LE.le ((OmegaCompletePartialOrder.fixedPoints.iterateChain (↑f) x h) n.succ) …
    -/
    apply le_ωSup
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      f : OmegaCompletePartialOrder.ContinuousHom α α
      x : α
      h : LE.le x (f x)
      ⊢ LE.le (OmegaCompletePartialOrder.ωSup (OmegaCompletePartialOrder.fixedPoints …
    -/
  · apply ωSup_le
    /-
      case a.a
      α : Type u_2
      inst✝ : OmegaCompletePartialOrder α
      f : OmegaCompletePartialOrder.ContinuousHom α α
      x : α
      h : LE.le x (f x)
      ⊢ ∀ (i : Nat), LE.le ((OmegaCompletePartialOrder.fixedPoints.iterateChain (↑f) …
    -/
    rintro (_ | n)
      /-
        case a.a.zero
        α : Type u_2
        inst✝ : OmegaCompletePartialOrder α
        f : OmegaCompletePartialOrder.ContinuousHom α α
        x : α
        h : LE.le x (f x)
        ⊢ LE.le ((OmegaCompletePartialOrder.fixedPoints.iterateChain (↑f) x h) 0) (Ome …
      -/
    · apply le_trans h
      /-
        case a.a.zero
        α : Type u_2
        inst✝ : OmegaCompletePartialOrder α
        f : OmegaCompletePartialOrder.ContinuousHom α α
        x : α
        h : LE.le x (f x)
        ⊢ LE.le (f x) (OmegaCompletePartialOrder.ωSup ((OmegaCompletePartialOrder.fixe …
      -/
      change ((iterateChain f x h).map f) 0 ≤ ωSup ((iterateChain f x h).map (f : α →o α))
      /-
        case a.a.zero
        α : Type u_2
        inst✝ : OmegaCompletePartialOrder α
        f : OmegaCompletePartialOrder.ContinuousHom α α
        x : α
        h : LE.le x (f x)
        ⊢ LE.le (((OmegaCompletePartialOrder.fixedPoints.iterateChain (↑f) x h).map ↑f …
      -/
      apply le_ωSup
      /-
        🎉 no goals
      -/
    · have : iterateChain f x h (n.succ) = (iterateChain f x h).map f n :=
        Function.iterate_succ_apply' ..
      /-
        case a.a.succ
        α : Type u_2
        inst✝ : OmegaCompletePartialOrder α
        f : OmegaCompletePartialOrder.ContinuousHom α α
        x : α
        h : LE.le x (f x)
        n : Nat
        this : Eq ((OmegaCompletePartialOrder.fixedPoints.iterateChain (↑f) x h) n.suc …
        ⊢ LE.le ((OmegaCompletePartialOrder.fixedPoints.iterateChain (↑f) x h) (HAdd.h …
      -/
      rw [this]
      /-
        case a.a.succ
        α : Type u_2
        inst✝ : OmegaCompletePartialOrder α
        f : OmegaCompletePartialOrder.ContinuousHom α α
        x : α
        h : LE.le x (f x)
        n : Nat
        this : Eq ((OmegaCompletePartialOrder.fixedPoints.iterateChain (↑f) x h) n.suc …
        ⊢ LE.le (((OmegaCompletePartialOrder.fixedPoints.iterateChain (↑f) x h).map ↑f …
      -/
      apply le_ωSup
      /-
        🎉 no goals
      -/


/-- The supremum of iterating a function on x arbitrary often is smaller than any prefixed point.

A prefixed point is a value `a` with `f a ≤ a`. -/
theorem ωSup_iterate_le_prefixedPoint (h : x ≤ f x) {a : α}
    (h_a : f a ≤ a) (h_x_le_a : x ≤ a) :
    ωSup (iterateChain f x h) ≤ a := by
  /-
    α : Type u_2
    inst✝ : OmegaCompletePartialOrder α
    f : OmegaCompletePartialOrder.ContinuousHom α α
    x : α
    h : LE.le x (f x)
    a : α
    h_a : LE.le (f a) a
    h_x_le_a : LE.le x a
    ⊢ LE.le (OmegaCompletePartialOrder.ωSup (OmegaCompletePartialOrder.fixedPoints …
  -/
  apply ωSup_le
  /-
    case a
    α : Type u_2
    inst✝ : OmegaCompletePartialOrder α
    f : OmegaCompletePartialOrder.ContinuousHom α α
    x : α
    h : LE.le x (f x)
    a : α
    h_a : LE.le (f a) a
    h_x_le_a : LE.le x a
    ⊢ ∀ (i : Nat), LE.le ((OmegaCompletePartialOrder.fixedPoints.iterateChain (↑f) …
  -/
  intro n
  induction n with
  | zero => exact h_x_le_a
  | succ n h_ind =>
    have : iterateChain f x h (n.succ) = f (iterateChain f x h n) :=
      Function.iterate_succ_apply' ..
    rw [this]
    exact le_trans (f.monotone h_ind) h_a


/-- The supremum of iterating a function on x arbitrary often is smaller than any fixed point. -/
theorem ωSup_iterate_le_fixedPoint (h : x ≤ f x) {a : α}
    (h_a : a ∈ fixedPoints f) (h_x_le_a : x ≤ a) :
    ωSup (iterateChain f x h) ≤ a := by
  /-
    α : Type u_2
    inst✝ : OmegaCompletePartialOrder α
    f : OmegaCompletePartialOrder.ContinuousHom α α
    x : α
    h : LE.le x (f x)
    a : α
    h_a : Membership.mem (Function.fixedPoints ⇑f) a
    h_x_le_a : LE.le x a
    ⊢ LE.le (OmegaCompletePartialOrder.ωSup (OmegaCompletePartialOrder.fixedPoints …
  -/
  rw [mem_fixedPoints] at h_a
  /-
    α : Type u_2
    inst✝ : OmegaCompletePartialOrder α
    f : OmegaCompletePartialOrder.ContinuousHom α α
    x : α
    h : LE.le x (f x)
    a : α
    h_a : Function.IsFixedPt (⇑f) a
    h_x_le_a : LE.le x a
    ⊢ LE.le (OmegaCompletePartialOrder.ωSup (OmegaCompletePartialOrder.fixedPoints …
  -/
  obtain h_a := Eq.le h_a
  /-
    α : Type u_2
    inst✝ : OmegaCompletePartialOrder α
    f : OmegaCompletePartialOrder.ContinuousHom α α
    x : α
    h : LE.le x (f x)
    a : α
    h_a✝ : Function.IsFixedPt (⇑f) a
    h_x_le_a : LE.le x a
    h_a : LE.le (f a) a
    ⊢ LE.le (OmegaCompletePartialOrder.ωSup (OmegaCompletePartialOrder.fixedPoints …
  -/
  exact ωSup_iterate_le_prefixedPoint f x h h_a h_x_le_a
  /-
    🎉 no goals
  -/


