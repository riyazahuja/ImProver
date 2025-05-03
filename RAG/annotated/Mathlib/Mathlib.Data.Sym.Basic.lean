/-- The nth symmetric power is n-tuples up to permutation.  We define it
as a subtype of `Multiset` since these are well developed in the
library.  We also give a definition `Sym.sym'` in terms of vectors, and we
show these are equivalent in `Sym.symEquivSym'`.
-/
def Sym (α : Type*) (n : ℕ) :=
  { s : Multiset α // Multiset.card s = n }

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11445): new definition

/-- The canonical map to `Multiset α` that forgets that `s` has length `n` -/
@[coe] def Sym.toMultiset {α : Type*} {n : ℕ} (s : Sym α n) : Multiset α :=
  s.1


instance Sym.hasCoe (α : Type*) (n : ℕ) : CoeOut (Sym α n) (Multiset α) :=
  ⟨Sym.toMultiset⟩

-- Porting note: instance needed for Data.Finset.Sym

instance {α : Type*} {n : ℕ} [DecidableEq α] : DecidableEq (Sym α n) :=
  inferInstanceAs <| DecidableEq <| Subtype _


/-- This is the `List.Perm` setoid lifted to `Vector`.

See note [reducible non-instances].
-/
abbrev List.Vector.Perm.isSetoid (α : Type*) (n : ℕ) : Setoid (Vector α n) :=
  (List.isSetoid α).comap Subtype.val


theorem coe_injective : Injective ((↑) : Sym α n → Multiset α) :=
  Subtype.coe_injective


@[simp, norm_cast]
theorem coe_inj {s₁ s₂ : Sym α n} : (s₁ : Multiset α) = s₂ ↔ s₁ = s₂ :=
  coe_injective.eq_iff


@[ext] theorem ext {s₁ s₂ : Sym α n} (h : (s₁ : Multiset α) = ↑s₂) : s₁ = s₂ :=
  coe_injective h


@[simp]
theorem val_eq_coe (s : Sym α n) : s.1 = ↑s :=
  rfl


/-- Construct an element of the `n`th symmetric power from a multiset of cardinality `n`.
-/
@[match_pattern] -- Porting note: removed `@[simps]`, generated bad lemma
abbrev mk (m : Multiset α) (h : Multiset.card m = n) : Sym α n :=
  ⟨m, h⟩


/-- The unique element in `Sym α 0`. -/
@[match_pattern]
def nil : Sym α 0 :=
  ⟨0, Multiset.card_zero⟩


@[simp]
theorem coe_nil : ↑(@Sym.nil α) = (0 : Multiset α) :=
  rfl


/-- Inserts an element into the term of `Sym α n`, increasing the length by one.
-/
@[match_pattern]
def cons (a : α) (s : Sym α n) : Sym α n.succ :=
                 /-
                   α : Type u_1
                   β : Type u_2
                   n n' m : Nat
                   s✝ : Sym α n
                   a✝ b a : α
                   s : Sym α n
                   ⊢ Eq (Multiset.cons a ↑s).card n.succ
                 -/
  ⟨a ::ₘ s.1, by rw [Multiset.card_cons, s.2]⟩
                 /-
                   🎉 no goals
                 -/


@[inherit_doc]
infixr:67 " ::ₛ " => cons


@[simp]
theorem cons_inj_right (a : α) (s s' : Sym α n) : a ::ₛ s = a ::ₛ s' ↔ s = s' :=
  Subtype.ext_iff.trans <| (Multiset.cons_inj_right _).trans Subtype.ext_iff.symm


@[simp]
theorem cons_inj_left (a a' : α) (s : Sym α n) : a ::ₛ s = a' ::ₛ s ↔ a = a' :=
  Subtype.ext_iff.trans <| Multiset.cons_inj_left _


theorem cons_swap (a b : α) (s : Sym α n) : a ::ₛ b ::ₛ s = b ::ₛ a ::ₛ s :=
  Subtype.ext <| Multiset.cons_swap a b s.1


theorem coe_cons (s : Sym α n) (a : α) : (a ::ₛ s : Multiset α) = a ::ₘ s :=
  rfl


/-- This is the quotient map that takes a list of n elements as an n-tuple and produces an nth
symmetric power.
-/
def ofVector : List.Vector α n → Sym α n :=
  fun x => ⟨↑x.val, (Multiset.coe_card _).trans x.2⟩


/-- This is the quotient map that takes a list of n elements as an n-tuple and produces an nth
symmetric power.
-/
instance : Coe (List.Vector α n) (Sym α n) where coe x := ofVector x


@[simp]
theorem ofVector_nil : ↑(Vector.nil : List.Vector α 0) = (Sym.nil : Sym α 0) :=
  rfl


@[simp]
theorem ofVector_cons (a : α) (v : List.Vector α n) :
    ↑(Vector.cons a v) = a ::ₛ (↑v : Sym α n) := by
  /-
    α : Type u_1
    n : Nat
    a : α
    v : List.Vector α n
    ⊢ Eq (Sym.ofVector (List.Vector.cons a v)) (Sym.cons a (Sym.ofVector v))
  -/
  cases v
  /-
    case mk
    α : Type u_1
    n : Nat
    a : α
    val✝ : List α
    property✝ : Eq val✝.length n
    ⊢ Eq (Sym.ofVector (List.Vector.cons a ⟨val✝, property✝⟩)) (Sym.cons a (Sym.of …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem card_coe : Multiset.card (s : Multiset α) = n := s.prop


/-- `α ∈ s` means that `a` appears as one of the factors in `s`.
-/
instance : Membership α (Sym α n) :=
  ⟨fun s a => a ∈ s.1⟩


instance decidableMem [DecidableEq α] (a : α) (s : Sym α n) : Decidable (a ∈ s) :=
  s.1.decidableMem _


@[simp, norm_cast] lemma coe_mk (s : Multiset α) (h : Multiset.card s = n) : mk s h = s := rfl


@[simp]
theorem mem_mk (a : α) (s : Multiset α) (h : Multiset.card s = n) : a ∈ mk s h ↔ a ∈ s :=
  Iff.rfl


lemma «forall» {p : Sym α n → Prop} :
    (∀ s : Sym α n, p s) ↔ ∀ (s : Multiset α) (hs : Multiset.card s = n), p (Sym.mk s hs) := by
  /-
    α : Type u_1
    n : Nat
    p : Sym α n → Prop
    ⊢ Iff (∀ (s : Sym α n), p s) (∀ (s : Multiset α) (hs : Eq s.card n), p (Sym.mk …
  -/
  simp [Sym]
  /-
    🎉 no goals
  -/


lemma «exists» {p : Sym α n → Prop} :
    (∃ s : Sym α n, p s) ↔ ∃ (s : Multiset α) (hs : Multiset.card s = n), p (Sym.mk s hs) := by
  /-
    α : Type u_1
    n : Nat
    p : Sym α n → Prop
    ⊢ Iff (Exists fun s => p s) (Exists fun s => Exists fun hs => p (Sym.mk s hs))
  -/
  simp [Sym]
  /-
    🎉 no goals
  -/


@[simp]
theorem not_mem_nil (a : α) : ¬ a ∈ (nil : Sym α 0) :=
  Multiset.not_mem_zero a


@[simp]
theorem mem_cons : a ∈ b ::ₛ s ↔ a = b ∨ a ∈ s :=
  Multiset.mem_cons


@[simp]
theorem mem_coe : a ∈ (s : Multiset α) ↔ a ∈ s :=
  Iff.rfl


theorem mem_cons_of_mem (h : a ∈ s) : a ∈ b ::ₛ s :=
  Multiset.mem_cons_of_mem h


theorem mem_cons_self (a : α) (s : Sym α n) : a ∈ a ::ₛ s :=
  Multiset.mem_cons_self a s.1


theorem cons_of_coe_eq (a : α) (v : List.Vector α n) : a ::ₛ (↑v : Sym α n) = ↑(a ::ᵥ v) :=
  Subtype.ext <| by
    /-
      α : Type u_1
      n : Nat
      a : α
      v : List.Vector α n
      ⊢ Eq ↑(Sym.cons a (Sym.ofVector v)) ↑(Sym.ofVector (List.Vector.cons a v))
    -/
    cases v
    /-
      case mk
      α : Type u_1
      n : Nat
      a : α
      val✝ : List α
      property✝ : Eq val✝.length n
      ⊢ Eq ↑(Sym.cons a (Sym.ofVector ⟨val✝, property✝⟩)) ↑(Sym.ofVector (List.Vecto …
    -/
    rfl
    /-
      🎉 no goals
    -/


open scoped List in
theorem sound {a b : List.Vector α n} (h : a.val ~ b.val) : (↑a : Sym α n) = ↑b :=
  Subtype.ext <| Quotient.sound h


/-- `erase s a h` is the sym that subtracts 1 from the
  multiplicity of `a` if a is present in the sym. -/
def erase [DecidableEq α] (s : Sym α (n + 1)) (a : α) (h : a ∈ s) : Sym α n :=
  ⟨s.val.erase a, (Multiset.card_erase_of_mem h).trans <| s.property.symm ▸ n.pred_succ⟩


@[simp]
theorem erase_mk [DecidableEq α] (m : Multiset α)
    (hc : Multiset.card m = n + 1) (a : α) (h : a ∈ m) :
    (mk m hc).erase a h =mk (m.erase a)
            /-
              α : Type u_1
              β : Type u_2
              n n' m✝ : Nat
              s : Sym α n
              a✝ b : α
              inst✝ : DecidableEq α
              m : Multiset α
              hc : Eq m.card (HAdd.hAdd n 1)
              a : α
              h : Membership.mem m a
              ⊢ Eq (m.erase a).card n
            -/
        (by rw [Multiset.card_erase_of_mem h, hc, Nat.add_one, Nat.pred_succ]) :=
            /-
              🎉 no goals
            -/
  rfl


@[simp]
theorem coe_erase [DecidableEq α] {s : Sym α n.succ} {a : α} (h : a ∈ s) :
    (s.erase a h : Multiset α) = Multiset.erase s a :=
  rfl


@[simp]
theorem cons_erase [DecidableEq α] {s : Sym α n.succ} {a : α} (h : a ∈ s) : a ::ₛ s.erase a h = s :=
  coe_injective <| Multiset.cons_erase h


@[simp]
theorem erase_cons_head [DecidableEq α] (s : Sym α n) (a : α)
    (h : a ∈ a ::ₛ s := mem_cons_self a s) : (a ::ₛ s).erase a h = s :=
  coe_injective <| Multiset.erase_cons_head a s.1


/-- Another definition of the nth symmetric power, using vectors modulo permutations. (See `Sym`.)
-/
def Sym' (α : Type*) (n : ℕ) :=
  Quotient (Vector.Perm.isSetoid α n)


/-- This is `cons` but for the alternative `Sym'` definition.
-/
def cons' {α : Type*} {n : ℕ} : α → Sym' α n → Sym' α (Nat.succ n) := fun a =>
  Quotient.map (Vector.cons a) fun ⟨_, _⟩ ⟨_, _⟩ h => List.Perm.cons _ h


@[inherit_doc]
scoped notation a " :: " b => cons' a b


/-- Multisets of cardinality n are equivalent to length-n vectors up to permutations.
-/
def symEquivSym' {α : Type*} {n : ℕ} : Sym α n ≃ Sym' α n :=
                                                             /-
                                                               α✝ : Type u_1
                                                               β : Type u_2
                                                               n✝ n' m : Nat
                                                               s : Sym α✝ n✝
                                                               a b : α✝
                                                               α : Type u_3
                                                               n : Nat
                                                               x✝ : List α
                                                               ⊢ Iff (Eq x✝.length n) (Eq (Multiset.card (Quotient.mk (List.isSetoid α) x✝)) n)
                                                             -/
                                                             /-
                                                               🎉 no goals
                                                             -/
  Equiv.subtypeQuotientEquivQuotientSubtype _ _ (fun _ => by rfl) fun _ _ => by rfl
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


theorem cons_equiv_eq_equiv_cons (α : Type*) (n : ℕ) (a : α) (s : Sym α n) :
    (a::symEquivSym' s) = symEquivSym' (a ::ₛ s) := by
  /-
    α : Type u_3
    n : Nat
    a : α
    s : Sym α n
    ⊢ Eq (Sym.cons' a (Sym.symEquivSym' s)) (Sym.symEquivSym' (Sym.cons a s))
  -/
  rcases s with ⟨⟨l⟩, _⟩
  /-
    case mk.mk
    α : Type u_3
    n : Nat
    a : α
    val✝ : Multiset α
    l : List α
    property✝ : Eq (Multiset.card (Quot.mk (⇑(List.isSetoid α)) l)) n
    ⊢ Eq (Sym.cons' a (Sym.symEquivSym' ⟨Quot.mk (⇑(List.isSetoid α)) l, property✝ …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance instZeroSym : Zero (Sym α 0) :=
  ⟨⟨0, rfl⟩⟩


@[simp] theorem toMultiset_zero : toMultiset (0 : Sym α 0) = 0 := rfl


instance : EmptyCollection (Sym α 0) :=
  ⟨0⟩


theorem eq_nil_of_card_zero (s : Sym α 0) : s = nil :=
  Subtype.ext <| Multiset.card_eq_zero.1 s.2


instance uniqueZero : Unique (Sym α 0) :=
  ⟨⟨nil⟩, eq_nil_of_card_zero⟩


/-- `replicate n a` is the sym containing only `a` with multiplicity `n`. -/
def replicate (n : ℕ) (a : α) : Sym α n :=
  ⟨Multiset.replicate n a, Multiset.card_replicate _ _⟩


theorem replicate_succ {a : α} {n : ℕ} : replicate n.succ a = a ::ₛ replicate n a :=
  rfl


theorem coe_replicate : (replicate n a : Multiset α) = Multiset.replicate n a :=
  rfl


@[simp]
theorem mem_replicate : b ∈ replicate n a ↔ n ≠ 0 ∧ b = a :=
  Multiset.mem_replicate


theorem eq_replicate_iff : s = replicate n a ↔ ∀ b ∈ s, b = a := by
  /-
    α : Type u_1
    n : Nat
    s : Sym α n
    a : α
    ⊢ Iff (Eq s (Sym.replicate n a)) (∀ (b : α), Membership.mem s b → Eq b a)
  -/
  erw [Subtype.ext_iff, Multiset.eq_replicate]
  /-
    α : Type u_1
    n : Nat
    s : Sym α n
    a : α
    ⊢ Iff (And (Eq (↑s).card n) (∀ (b : α), Membership.mem (↑s) b → Eq b a)) (∀ (b …
  -/
  exact and_iff_right s.2
  /-
    🎉 no goals
  -/


theorem exists_mem (s : Sym α n.succ) : ∃ a, a ∈ s :=
  Multiset.card_pos_iff_exists_mem.1 <| s.2.symm ▸ n.succ_pos


theorem exists_cons_of_mem {s : Sym α (n + 1)} {a : α} (h : a ∈ s) : ∃ t, s = a ::ₛ t := by
  /-
    α : Type u_1
    n : Nat
    s : Sym α (HAdd.hAdd n 1)
    a : α
    h : Membership.mem s a
    ⊢ Exists fun t => Eq s (Sym.cons a t)
  -/
  obtain ⟨m, h⟩ := Multiset.exists_cons_of_mem h
  have : Multiset.card m = n := by
    apply_fun Multiset.card at h
    rw [s.2, Multiset.card_cons, add_left_inj] at h
    exact h.symm
  /-
    case intro
    α : Type u_1
    n : Nat
    s : Sym α (HAdd.hAdd n 1)
    a : α
    h✝ : Membership.mem s a
    m : Multiset α
    h : Eq (↑s) (Multiset.cons a m)
    this : Eq m.card n
    ⊢ Exists fun t => Eq s (Sym.cons a t)
  -/
  use ⟨m, this⟩
  /-
    case h
    α : Type u_1
    n : Nat
    s : Sym α (HAdd.hAdd n 1)
    a : α
    h✝ : Membership.mem s a
    m : Multiset α
    h : Eq (↑s) (Multiset.cons a m)
    this : Eq m.card n
    ⊢ Eq s (Sym.cons a ⟨m, this⟩)
  -/
  apply Subtype.ext
  /-
    case h.a
    α : Type u_1
    n : Nat
    s : Sym α (HAdd.hAdd n 1)
    a : α
    h✝ : Membership.mem s a
    m : Multiset α
    h : Eq (↑s) (Multiset.cons a m)
    this : Eq m.card n
    ⊢ Eq ↑s ↑(Sym.cons a ⟨m, this⟩)
  -/
  exact h
  /-
    🎉 no goals
  -/


theorem exists_eq_cons_of_succ (s : Sym α n.succ) : ∃ (a : α) (s' : Sym α n), s = a ::ₛ s' := by
  /-
    α : Type u_1
    n : Nat
    s : Sym α n.succ
    ⊢ Exists fun a => Exists fun s' => Eq s (Sym.cons a s')
  -/
  obtain ⟨a, ha⟩ := exists_mem s
  /-
    case intro
    α : Type u_1
    n : Nat
    s : Sym α n.succ
    a : α
    ha : Membership.mem s a
    ⊢ Exists fun a => Exists fun s' => Eq s (Sym.cons a s')
  -/
  classical exact ⟨a, s.erase a ha, (cons_erase ha).symm⟩
  /-
    🎉 no goals
  -/


theorem eq_replicate {a : α} {n : ℕ} {s : Sym α n} : s = replicate n a ↔ ∀ b ∈ s, b = a :=
  Subtype.ext_iff.trans <| Multiset.eq_replicate.trans <| and_iff_right s.prop


theorem eq_replicate_of_subsingleton [Subsingleton α] (a : α) {n : ℕ} (s : Sym α n) :
    s = replicate n a :=
  eq_replicate.2 fun _ _ => Subsingleton.elim _ _


instance [Subsingleton α] (n : ℕ) : Subsingleton (Sym α n) :=
  ⟨by
    /-
      α : Type u_1
      β : Type u_2
      n✝ n' m : Nat
      s : Sym α n✝
      a b : α
      inst✝ : Subsingleton α
      n : Nat
      ⊢ ∀ (a b : Sym α n), Eq a b
    -/
    cases n
      /-
        case zero
        α : Type u_1
        β : Type u_2
        n n' m : Nat
        s : Sym α n
        a b : α
        inst✝ : Subsingleton α
        ⊢ ∀ (a b : Sym α 0), Eq a b
      -/
    · simp [eq_iff_true_of_subsingleton]
      /-
        🎉 no goals
      -/
      /-
        case succ
        α : Type u_1
        β : Type u_2
        n n' m : Nat
        s : Sym α n
        a b : α
        inst✝ : Subsingleton α
        n✝ : Nat
        ⊢ ∀ (a b : Sym α (HAdd.hAdd n✝ 1)), Eq a b
      -/
    · intro s s'
      /-
        case succ
        α : Type u_1
        β : Type u_2
        n n' m : Nat
        s✝ : Sym α n
        a b : α
        inst✝ : Subsingleton α
        n✝ : Nat
        s s' : Sym α (HAdd.hAdd n✝ 1)
        ⊢ Eq s s'
      -/
      obtain ⟨b, -⟩ := exists_mem s
      /-
        case succ.intro
        α : Type u_1
        β : Type u_2
        n n' m : Nat
        s✝ : Sym α n
        a b✝ : α
        inst✝ : Subsingleton α
        n✝ : Nat
        s s' : Sym α (HAdd.hAdd n✝ 1)
        b : α
        ⊢ Eq s s'
      -/
      rw [eq_replicate_of_subsingleton b s', eq_replicate_of_subsingleton b s]⟩
      /-
        🎉 no goals
      -/


instance inhabitedSym [Inhabited α] (n : ℕ) : Inhabited (Sym α n) :=
  ⟨replicate n default⟩


instance inhabitedSym' [Inhabited α] (n : ℕ) : Inhabited (Sym' α n) :=
  ⟨Quotient.mk' (Vector.replicate n default)⟩


instance (n : ℕ) [IsEmpty α] : IsEmpty (Sym α n.succ) :=
  ⟨fun s => by
    /-
      α : Type u_1
      β : Type u_2
      n✝ n' m : Nat
      s✝ : Sym α n✝
      a b : α
      n : Nat
      inst✝ : IsEmpty α
      s : Sym α n.succ
      ⊢ False
    -/
    obtain ⟨a, -⟩ := exists_mem s
    /-
      case intro
      α : Type u_1
      β : Type u_2
      n✝ n' m : Nat
      s✝ : Sym α n✝
      a✝ b : α
      n : Nat
      inst✝ : IsEmpty α
      s : Sym α n.succ
      a : α
      ⊢ False
    -/
    exact isEmptyElim a⟩
    /-
      🎉 no goals
    -/


instance (n : ℕ) [Unique α] : Unique (Sym α n) :=
  Unique.mk' _


theorem replicate_right_inj {a b : α} {n : ℕ} (h : n ≠ 0) : replicate n a = replicate n b ↔ a = b :=
  Subtype.ext_iff.trans (Multiset.replicate_right_inj h)


theorem replicate_right_injective {n : ℕ} (h : n ≠ 0) :
    Function.Injective (replicate n : α → Sym α n) := fun _ _ => (replicate_right_inj h).1


instance (n : ℕ) [Nontrivial α] : Nontrivial (Sym α (n + 1)) :=
  (replicate_right_injective n.succ_ne_zero).nontrivial


/-- A function `α → β` induces a function `Sym α n → Sym β n` by applying it to every element of
the underlying `n`-tuple. -/
def map {n : ℕ} (f : α → β) (x : Sym α n) : Sym β n :=
                   /-
                     α : Type u_1
                     β : Type u_2
                     n✝ n' m : Nat
                     s : Sym α n✝
                     a b : α
                     n : Nat
                     f : α → β
                     x : Sym α n
                     ⊢ Eq (Multiset.map f ↑x).card n
                   -/
  ⟨x.val.map f, by simp⟩
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem mem_map {n : ℕ} {f : α → β} {b : β} {l : Sym α n} :
    b ∈ Sym.map f l ↔ ∃ a, a ∈ l ∧ f a = b :=
  Multiset.mem_map


/-- Note: `Sym.map_id` is not simp-normal, as simp ends up unfolding `id` with `Sym.map_congr` -/
@[simp]
theorem map_id' {α : Type*} {n : ℕ} (s : Sym α n) : Sym.map (fun x : α => x) s = s := by
  /-
    α : Type u_3
    n : Nat
    s : Sym α n
    ⊢ Eq (Sym.map (fun x => x) s) s
  -/
  ext; simp only [map, Multiset.map_id', ← val_eq_coe]
       /-
         🎉 no goals
       -/


theorem map_id {α : Type*} {n : ℕ} (s : Sym α n) : Sym.map id s = s := by
  /-
    α : Type u_3
    n : Nat
    s : Sym α n
    ⊢ Eq (Sym.map id s) s
  -/
  ext; simp only [map, id_eq, Multiset.map_id', ← val_eq_coe]
       /-
         🎉 no goals
       -/


@[simp]
theorem map_map {α β γ : Type*} {n : ℕ} (g : β → γ) (f : α → β) (s : Sym α n) :
    Sym.map g (Sym.map f s) = Sym.map (g ∘ f) s :=
                    /-
                      α : Type u_3
                      β : Type u_4
                      γ : Type u_5
                      n : Nat
                      g : β → γ
                      f : α → β
                      s : Sym α n
                      ⊢ Eq ↑(Sym.map g (Sym.map f s)) ↑(Sym.map (Function.comp g f) s)
                    -/
  Subtype.ext <| by dsimp only [Sym.map]; simp
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
theorem map_zero (f : α → β) : Sym.map f (0 : Sym α 0) = (0 : Sym β 0) :=
  rfl


@[simp]
theorem map_cons {n : ℕ} (f : α → β) (a : α) (s : Sym α n) : (a ::ₛ s).map f = f a ::ₛ s.map f :=
  ext <| Multiset.map_cons _ _ _


@[congr]
theorem map_congr {f g : α → β} {s : Sym α n} (h : ∀ x ∈ s, f x = g x) : map f s = map g s :=
  Subtype.ext <| Multiset.map_congr rfl h


@[simp]
theorem map_mk {f : α → β} {m : Multiset α} {hc : Multiset.card m = n} :
                                       /-
                                         α : Type u_1
                                         β : Type u_2
                                         n n' m✝ : Nat
                                         s : Sym α n
                                         a b : α
                                         f : α → β
                                         m : Multiset α
                                         hc : Eq m.card n
                                         ⊢ Eq (Multiset.map f m).card n
                                       -/
    map f (mk m hc) = mk (m.map f) (by simp [hc]) :=
                                       /-
                                         🎉 no goals
                                       -/
  rfl


@[simp]
theorem coe_map (s : Sym α n) (f : α → β) : ↑(s.map f) = Multiset.map f s :=
  rfl


theorem map_injective {f : α → β} (hf : Injective f) (n : ℕ) :
    Injective (map f : Sym α n → Sym β n) := fun _ _ h =>
  coe_injective <| Multiset.map_injective hf <| coe_inj.2 h


/-- Mapping an equivalence `α ≃ β` using `Sym.map` gives an equivalence between `Sym α n` and
`Sym β n`. -/
@[simps]
def equivCongr (e : α ≃ β) : Sym α n ≃ Sym β n where
  toFun := map e
  invFun := map e.symm
                   /-
                     α : Type u_1
                     β : Type u_2
                     n n' m : Nat
                     s : Sym α n
                     a b : α
                     e : Equiv α β
                     x : Sym α n
                     ⊢ Eq (Sym.map (⇑e.symm) (Sym.map (⇑e) x)) x
                   -/
  left_inv x := by rw [map_map, Equiv.symm_comp_self, map_id]
                   /-
                     🎉 no goals
                   -/
                    /-
                      α : Type u_1
                      β : Type u_2
                      n n' m : Nat
                      s : Sym α n
                      a b : α
                      e : Equiv α β
                      x : Sym β n
                      ⊢ Eq (Sym.map (⇑e) (Sym.map (⇑e.symm) x)) x
                    -/
  right_inv x := by rw [map_map, Equiv.self_comp_symm, map_id]
                    /-
                      🎉 no goals
                    -/


/-- "Attach" a proof that `a ∈ s` to each element `a` in `s` to produce
an element of the symmetric power on `{x // x ∈ s}`. -/
def attach (s : Sym α n) : Sym { x // x ∈ s } n :=
                     /-
                       α : Type u_1
                       β : Type u_2
                       n n' m : Nat
                       s✝ : Sym α n
                       a b : α
                       s : Sym α n
                       ⊢ Eq (↑s).attach.card n
                     -/
  ⟨s.val.attach, by (conv_rhs => rw [← s.2, ← Multiset.card_attach])⟩
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem attach_mk {m : Multiset α} {hc : Multiset.card m = n} :
    attach (mk m hc) = mk m.attach (Multiset.card_attach.trans hc) :=
  rfl


@[simp]
theorem coe_attach (s : Sym α n) : (s.attach : Multiset { a // a ∈ s }) =
    Multiset.attach (s : Multiset α) :=
  rfl


theorem attach_map_coe (s : Sym α n) : s.attach.map (↑) = s :=
  coe_injective <| Multiset.attach_map_val _


@[simp]
theorem mem_attach (s : Sym α n) (x : { x // x ∈ s }) : x ∈ s.attach :=
  Multiset.mem_attach _ _


@[simp]
theorem attach_nil : (nil : Sym α 0).attach = nil :=
  rfl


@[simp]
theorem attach_cons (x : α) (s : Sym α n) :
    (cons x s).attach =
      cons ⟨x, mem_cons_self _ _⟩ (s.attach.map fun x => ⟨x, mem_cons_of_mem x.prop⟩) :=
  coe_injective <| Multiset.attach_cons _ _


/-- Change the length of a `Sym` using an equality.
The simp-normal form is for the `cast` to be pushed outward. -/
protected def cast {n m : ℕ} (h : n = m) : Sym α n ≃ Sym α m where
  toFun s := ⟨s.val, s.2.trans h⟩
  invFun s := ⟨s.val, s.2.trans h.symm⟩
  left_inv _ := Subtype.ext rfl
  right_inv _ := Subtype.ext rfl


@[simp]
theorem cast_rfl : Sym.cast rfl s = s :=
  Subtype.ext rfl


@[simp]
theorem cast_cast {n'' : ℕ} (h : n = n') (h' : n' = n'') :
    Sym.cast h' (Sym.cast h s) = Sym.cast (h.trans h') s :=
  rfl


@[simp]
theorem coe_cast (h : n = m) : (Sym.cast h s : Multiset α) = s :=
  rfl


@[simp]
theorem mem_cast (h : n = m) : a ∈ Sym.cast h s ↔ a ∈ s :=
  Iff.rfl


/-- Append a pair of `Sym` terms. -/
def append (s : Sym α n) (s' : Sym α n') : Sym α (n + n') :=
                  /-
                    α : Type u_1
                    β : Type u_2
                    n n' m : Nat
                    s✝ : Sym α n
                    a b : α
                    s : Sym α n
                    s' : Sym α n'
                    ⊢ Eq (HAdd.hAdd ↑s ↑s').card (HAdd.hAdd n n')
                  -/
  ⟨s.1 + s'.1, by rw [Multiset.card_add, s.2, s'.2]⟩
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem append_inj_right (s : Sym α n) {t t' : Sym α n'} : s.append t = s.append t' ↔ t = t' :=
  Subtype.ext_iff.trans <| (add_right_inj _).trans Subtype.ext_iff.symm


@[simp]
theorem append_inj_left {s s' : Sym α n} (t : Sym α n') : s.append t = s'.append t ↔ s = s' :=
  Subtype.ext_iff.trans <| (add_left_inj _).trans Subtype.ext_iff.symm


theorem append_comm (s : Sym α n') (s' : Sym α n') :
    s.append s' = Sym.cast (add_comm _ _) (s'.append s) := by
  /-
    α : Type u_1
    n' : Nat
    s s' : Sym α n'
    ⊢ Eq (s.append s') ((Sym.cast ⋯) (s'.append s))
  -/
  ext
  /-
    case h
    α : Type u_1
    n' : Nat
    s s' : Sym α n'
    ⊢ Eq ↑(s.append s') ↑((Sym.cast ⋯) (s'.append s))
  -/
  simp [append, add_comm]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_append (s : Sym α n) (s' : Sym α n') : (s.append s' : Multiset α) = s + s' :=
  rfl


theorem mem_append_iff {s' : Sym α m} : a ∈ s.append s' ↔ a ∈ s ∨ a ∈ s' :=
  Multiset.mem_add


/-- `a ↦ {a}` as an equivalence between `α` and `Sym α 1`. -/
@[simps apply]
def oneEquiv : α ≃ Sym α 1 where
                      /-
                        α : Type u_1
                        β : Type u_2
                        n n' m : Nat
                        s : Sym α n
                        a✝ b a : α
                        ⊢ Eq (Singleton.singleton a).card 1
                      -/
  toFun a := ⟨{a}, by simp⟩
                      /-
                        🎉 no goals
                      -/
  invFun s := (Equiv.subtypeQuotientEquivQuotientSubtype
                                                        /-
                                                          α : Type u_1
                                                          β : Type u_2
                                                          n n' m : Nat
                                                          s✝ : Sym α n
                                                          a b : α
                                                          s : Sym α 1
                                                          l l' : Subtype fun x => Eq x.length 1
                                                          ⊢ Iff ((?m.30256 s) l l') ((List.isSetoid α) ↑l ↑l')
                                                        -/
      (·.length = 1) _ (fun _ ↦ Iff.rfl) (fun l l' ↦ by rfl) s).liftOn
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                  /-
                                                    α : Type u_1
                                                    β : Type u_2
                                                    n n' m : Nat
                                                    s✝ : Sym α n
                                                    a b : α
                                                    s : Sym α 1
                                                    l : Subtype fun x => Eq x.length 1
                                                    ⊢ LT.lt 0 (↑l).length
                                                  -/
    (fun l ↦ l.1.head <| List.length_pos.mp <| by simp)
                                                  /-
                                                    🎉 no goals
                                                  -/
    fun ⟨_, _⟩ ⟨_, h⟩ ↦ fun perm ↦ by
      /-
        α : Type u_1
        β : Type u_2
        n n' m : Nat
        s✝ : Sym α n
        a b : α
        s : Sym α 1
        x✝¹ x✝ : Subtype fun x => Eq x.length 1
        val✝¹ : List α
        property✝ : Eq val✝¹.length 1
        val✝ : List α
        h : Eq val✝.length 1
        perm : HasEquiv.Equiv ⟨val✝¹, property✝⟩ ⟨val✝, h⟩
        ⊢ Eq ((fun l => (↑l).head ⋯) ⟨val✝¹, property✝⟩) ((fun l => (↑l).head ⋯) ⟨val✝ …
      -/
      obtain ⟨a, rfl⟩ := List.length_eq_one.mp h
      /-
        case intro
        α : Type u_1
        β : Type u_2
        n n' m : Nat
        s✝ : Sym α n
        a✝ b : α
        s : Sym α 1
        x✝¹ x✝ : Subtype fun x => Eq x.length 1
        val✝ : List α
        property✝ : Eq val✝.length 1
        a : α
        h : Eq (List.cons a List.nil).length 1
        perm : HasEquiv.Equiv ⟨val✝, property✝⟩ ⟨List.cons a List.nil, h⟩
        ⊢ Eq ((fun l => (↑l).head ⋯) ⟨val✝, property✝⟩) ((fun l => (↑l).head ⋯) ⟨List. …
      -/
      exact List.eq_of_mem_singleton (perm.mem_iff.mp <| List.head_mem _)
      /-
        🎉 no goals
      -/
                   /-
                     α : Type u_1
                     β : Type u_2
                     n n' m : Nat
                     s : Sym α n
                     a✝ b a : α
                     ⊢ Eq ((fun s => ((Equiv.subtypeQuotientEquivQuotientSubtype (fun x => Eq x.len …
                   -/
  left_inv a := by rfl
                   /-
                     🎉 no goals
                   -/
                  /-
                    α : Type u_1
                    β : Type u_2
                    n n' m : Nat
                    s : Sym α n
                    a b : α
                    ⊢ Function.RightInverse (fun s => ((Equiv.subtypeQuotientEquivQuotientSubtype  …
                  -/
  right_inv := by rintro ⟨⟨l⟩, h⟩; obtain ⟨a, rfl⟩ := List.length_eq_one.mp h; rfl
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- Fill a term `m : Sym α (n - i)` with `i` copies of `a` to obtain a term of `Sym α n`.
This is a convenience wrapper for `m.append (replicate i a)` that adjusts the term using
`Sym.cast`. -/
def fill (a : α) (i : Fin (n + 1)) (m : Sym α (n - i)) : Sym α n :=
  Sym.cast (Nat.sub_add_cancel i.is_le) (m.append (replicate i a))


theorem coe_fill {a : α} {i : Fin (n + 1)} {m : Sym α (n - i)} :
    (fill a i m : Multiset α) = m + replicate i a :=
  rfl


theorem mem_fill_iff {a b : α} {i : Fin (n + 1)} {s : Sym α (n - i)} :
    a ∈ Sym.fill b i s ↔ (i : ℕ) ≠ 0 ∧ a = b ∨ a ∈ s := by
  /-
    α : Type u_1
    n : Nat
    a b : α
    i : Fin (HAdd.hAdd n 1)
    s : Sym α (HSub.hSub n ↑i)
    ⊢ Iff (Membership.mem (Sym.fill b i s) a) (Or (And (Ne (↑i) 0) (Eq a b)) (Memb …
  -/
  rw [fill, mem_cast, mem_append_iff, or_comm, mem_replicate]
  /-
    🎉 no goals
  -/


/-- Remove every `a` from a given `Sym α n`.
Yields the number of copies `i` and a term of `Sym α (n - i)`. -/
def filterNe [DecidableEq α] (a : α) (m : Sym α n) : Σi : Fin (n + 1), Sym α (n - i) :=
                                                    /-
                                                      α : Type u_1
                                                      β : Type u_2
                                                      n n' m✝ : Nat
                                                      s : Sym α n
                                                      a✝ b : α
                                                      inst✝ : DecidableEq α
                                                      a : α
                                                      m : Sym α n
                                                      ⊢ LT.lt (↑m).card (HAdd.hAdd n 1)
                                                    -/
  ⟨⟨m.1.count a, (count_le_card _ _).trans_lt <| by rw [m.2, Nat.lt_succ_iff]⟩,
                                                    /-
                                                      🎉 no goals
                                                    -/
    m.1.filter (a ≠ ·),
    Nat.eq_sub_of_add_eq <|
      Eq.trans
        (by
          /-
            α : Type u_1
            β : Type u_2
            n n' m✝ : Nat
            s : Sym α n
            a✝ b : α
            inst✝ : DecidableEq α
            a : α
            m : Sym α n
            ⊢ Eq (HAdd.hAdd (Multiset.filter (fun x => Ne a x) ↑m).card ↑⟨Multiset.count a …
          -/
          rw [← countP_eq_card_filter, add_comm]
          /-
            α : Type u_1
            β : Type u_2
            n n' m✝ : Nat
            s : Sym α n
            a✝ b : α
            inst✝ : DecidableEq α
            a : α
            m : Sym α n
            ⊢ Eq (HAdd.hAdd (↑⟨Multiset.count a ↑m, ⋯⟩) (Multiset.countP (fun x => Ne a x) …
          -/
          simp only [eq_comm, Ne, count]
          /-
            α : Type u_1
            β : Type u_2
            n n' m✝ : Nat
            s : Sym α n
            a✝ b : α
            inst✝ : DecidableEq α
            a : α
            m : Sym α n
            ⊢ Eq (↑m).card (HAdd.hAdd (Multiset.countP (fun x => Eq a x) ↑m) (Multiset.cou …
          -/
          rw [← card_eq_countP_add_countP _ _])
          /-
            🎉 no goals
          -/
        m.2⟩


theorem sigma_sub_ext {m₁ m₂ : Σi : Fin (n + 1), Sym α (n - i)} (h : (m₁.2 : Multiset α) = m₂.2) :
    m₁ = m₂ :=
  Sigma.subtype_ext
    (Fin.ext <| by
      rw [← Nat.sub_sub_self (Nat.le_of_lt_succ m₁.1.is_lt), ← m₁.2.2, val_eq_coe, h,
        ← val_eq_coe, m₂.2.2, Nat.sub_sub_self (Nat.le_of_lt_succ m₂.1.is_lt)])
    h


theorem fill_filterNe [DecidableEq α] (a : α) (m : Sym α n) :
    (m.filterNe a).2.fill a (m.filterNe a).1 = m :=
  Sym.ext
    (by
      /-
        α : Type u_1
        n : Nat
        inst✝ : DecidableEq α
        a : α
        m : Sym α n
        ⊢ Eq ↑(Sym.fill a (Sym.filterNe a m).fst (Sym.filterNe a m).snd) ↑m
      -/
      rw [coe_fill, filterNe, ← val_eq_coe, Subtype.coe_mk, Fin.val_mk]
      /-
        α : Type u_1
        n : Nat
        inst✝ : DecidableEq α
        a : α
        m : Sym α n
        ⊢ Eq (HAdd.hAdd (Multiset.filter (fun x => Ne a x) ↑m) ↑(Sym.replicate (Multis …
      -/
      ext b; dsimp
      /-
        case a
        α : Type u_1
        n : Nat
        inst✝ : DecidableEq α
        a : α
        m : Sym α n
        b : α
        ⊢ Eq (Multiset.count b (HAdd.hAdd (Multiset.filter (fun x => Not (Eq a x)) ↑m) …
      -/
      rw [count_add, count_filter, Sym.coe_replicate, count_replicate]
      /-
        case a
        α : Type u_1
        n : Nat
        inst✝ : DecidableEq α
        a : α
        m : Sym α n
        b : α
        ⊢ Eq (HAdd.hAdd (ite (Not (Eq a b)) (Multiset.count b ↑m) 0) (ite (Eq a b) (Mu …
      -/
      obtain rfl | h := eq_or_ne a b
        /-
          case a.inl
          α : Type u_1
          n : Nat
          inst✝ : DecidableEq α
          a : α
          m : Sym α n
          ⊢ Eq (HAdd.hAdd (ite (Not (Eq a a)) (Multiset.count a ↑m) 0) (ite (Eq a a) (Mu …
        -/
      · rw [if_pos rfl, if_neg (not_not.2 rfl), zero_add]
        /-
          🎉 no goals
        -/
        /-
          case a.inr
          α : Type u_1
          n : Nat
          inst✝ : DecidableEq α
          a : α
          m : Sym α n
          b : α
          h : Ne a b
          ⊢ Eq (HAdd.hAdd (ite (Not (Eq a b)) (Multiset.count b ↑m) 0) (ite (Eq a b) (Mu …
        -/
      · rw [if_pos h, if_neg h, add_zero])
        /-
          🎉 no goals
        -/


theorem filter_ne_fill [DecidableEq α] (a : α) (m : Σi : Fin (n + 1), Sym α (n - i)) (h : a ∉ m.2) :
    (m.2.fill a m.1).filterNe a = m :=
  sigma_sub_ext
    (by
      /-
        α : Type u_1
        n : Nat
        inst✝ : DecidableEq α
        a : α
        m : Sigma fun i => Sym α (HSub.hSub n ↑i)
        h : Not (Membership.mem m.snd a)
        ⊢ Eq ↑(Sym.filterNe a (Sym.fill a m.fst m.snd)).snd ↑m.snd
      -/
      rw [filterNe, ← val_eq_coe, Subtype.coe_mk, val_eq_coe, coe_fill]
      /-
        α : Type u_1
        n : Nat
        inst✝ : DecidableEq α
        a : α
        m : Sigma fun i => Sym α (HSub.hSub n ↑i)
        h : Not (Membership.mem m.snd a)
        ⊢ Eq (Multiset.filter (fun x => Ne a x) (HAdd.hAdd ↑m.snd ↑(Sym.replicate (↑m. …
      -/
      rw [filter_add, filter_eq_self.2, add_right_eq_self, eq_zero_iff_forall_not_mem]
        /-
          α : Type u_1
          n : Nat
          inst✝ : DecidableEq α
          a : α
          m : Sigma fun i => Sym α (HSub.hSub n ↑i)
          h : Not (Membership.mem m.snd a)
          ⊢ ∀ (a_1 : α), Not (Membership.mem (Multiset.filter (fun x => Ne a x) ↑(Sym.re …
        -/
      · intro b hb
        /-
          α : Type u_1
          n : Nat
          inst✝ : DecidableEq α
          a : α
          m : Sigma fun i => Sym α (HSub.hSub n ↑i)
          h : Not (Membership.mem m.snd a)
          b : α
          hb : Membership.mem (Multiset.filter (fun x => Ne a x) ↑(Sym.replicate (↑m.fst …
          ⊢ False
        -/
        rw [mem_filter, Sym.mem_coe, mem_replicate] at hb
        /-
          α : Type u_1
          n : Nat
          inst✝ : DecidableEq α
          a : α
          m : Sigma fun i => Sym α (HSub.hSub n ↑i)
          h : Not (Membership.mem m.snd a)
          b : α
          hb : And (And (Ne (↑m.fst) 0) (Eq b a)) (Ne a b)
          ⊢ False
        -/
        exact hb.2 hb.1.2.symm
        /-
          🎉 no goals
        -/
        /-
          α : Type u_1
          n : Nat
          inst✝ : DecidableEq α
          a : α
          m : Sigma fun i => Sym α (HSub.hSub n ↑i)
          h : Not (Membership.mem m.snd a)
          ⊢ ∀ (a_1 : α), Membership.mem (↑m.snd) a_1 → Ne a a_1
        -/
      · exact fun a ha ha' => h <| ha'.symm ▸ ha)
        /-
          🎉 no goals
        -/


theorem count_coe_fill_self_of_not_mem [DecidableEq α] {a : α} {i : Fin (n + 1)} {s : Sym α (n - i)}
    (hx : a ∉ s) :
    count a (fill a i s : Multiset α) = i := by
  /-
    α : Type u_1
    n : Nat
    inst✝ : DecidableEq α
    a : α
    i : Fin (HAdd.hAdd n 1)
    s : Sym α (HSub.hSub n ↑i)
    hx : Not (Membership.mem s a)
    ⊢ Eq (Multiset.count a ↑(Sym.fill a i s)) ↑i
  -/
  simp [coe_fill, coe_replicate, hx]
  /-
    🎉 no goals
  -/


theorem count_coe_fill_of_ne [DecidableEq α] {a x : α} {i : Fin (n + 1)} {s : Sym α (n - i)}
    (hx : x ≠ a) :
    count x (fill a i s : Multiset α) = count x s := by
  /-
    α : Type u_1
    n : Nat
    inst✝ : DecidableEq α
    a x : α
    i : Fin (HAdd.hAdd n 1)
    s : Sym α (HSub.hSub n ↑i)
    hx : Ne x a
    ⊢ Eq (Multiset.count x ↑(Sym.fill a i s)) (Multiset.count x ↑s)
  -/
  suffices x ∉ Multiset.replicate i a by simp [coe_fill, coe_replicate, this]
  /-
    α : Type u_1
    n : Nat
    inst✝ : DecidableEq α
    a x : α
    i : Fin (HAdd.hAdd n 1)
    s : Sym α (HSub.hSub n ↑i)
    hx : Ne x a
    ⊢ Not (Membership.mem (Multiset.replicate (↑i) a) x)
  -/
  simp [Multiset.mem_replicate, hx]
  /-
    🎉 no goals
  -/


/-- Function from the symmetric product over `Option` splitting on whether or not
it contains a `none`. -/
def encode [DecidableEq α] (s : Sym (Option α) n.succ) : Sym (Option α) n ⊕ Sym α n.succ :=
  if h : none ∈ s then Sum.inl (s.erase none h)
  else
    Sum.inr
      (s.attach.map fun o =>
        o.1.get <| Option.ne_none_iff_isSome.1 <| ne_of_mem_of_not_mem o.2 h)


@[simp]
theorem encode_of_none_mem [DecidableEq α] (s : Sym (Option α) n.succ) (h : none ∈ s) :
    encode s = Sum.inl (s.erase none h) :=
  dif_pos h


@[simp]
theorem encode_of_not_none_mem [DecidableEq α] (s : Sym (Option α) n.succ) (h : ¬none ∈ s) :
    encode s =
      Sum.inr
        (s.attach.map fun o =>
          o.1.get <| Option.ne_none_iff_isSome.1 <| ne_of_mem_of_not_mem o.2 h) :=
  dif_neg h


/-- Inverse of `Sym_option_succ_equiv.decode`. -/
-- @[simp] Porting note: not a nice simp lemma, applies too often in Lean4
def decode : Sym (Option α) n ⊕ Sym α n.succ → Sym (Option α) n.succ
  | Sum.inl s => none ::ₛ s
  | Sum.inr s => s.map Embedding.some


@[simp]
theorem decode_inl (s : Sym (Option α) n) : decode (Sum.inl s) = none ::ₛ s :=
  rfl


@[simp]
theorem decode_inr (s : Sym α n.succ) : decode (Sum.inr s) = s.map Embedding.some :=
  rfl


@[simp]
theorem decode_encode [DecidableEq α] (s : Sym (Option α) n.succ) : decode (encode s) = s := by
  /-
    α : Type u_1
    n : Nat
    inst✝ : DecidableEq α
    s : Sym (Option α) n.succ
    ⊢ Eq (SymOptionSuccEquiv.decode (SymOptionSuccEquiv.encode s)) s
  -/
  by_cases h : none ∈ s
    /-
      case pos
      α : Type u_1
      n : Nat
      inst✝ : DecidableEq α
      s : Sym (Option α) n.succ
      h : Membership.mem s Option.none
      ⊢ Eq (SymOptionSuccEquiv.decode (SymOptionSuccEquiv.encode s)) s
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
  · simp only [decode, h, not_false_iff, encode_of_not_none_mem, Embedding.some_apply, map_map,
      comp_apply, Option.some_get]
    /-
      case neg
      α : Type u_1
      n : Nat
      inst✝ : DecidableEq α
      s : Sym (Option α) n.succ
      h : Not (Membership.mem s Option.none)
      ⊢ Eq (Sym.map (fun x => ↑x) s.attach) s
    -/
    convert s.attach_map_coe
    /-
      🎉 no goals
    -/


@[simp]
theorem encode_decode [DecidableEq α] (s : Sym (Option α) n ⊕ Sym α n.succ) :
    encode (decode s) = s := by
  /-
    α : Type u_1
    n : Nat
    inst✝ : DecidableEq α
    s : Sum (Sym (Option α) n) (Sym α n.succ)
    ⊢ Eq (SymOptionSuccEquiv.encode (SymOptionSuccEquiv.decode s)) s
  -/
  obtain s | s := s
    /-
      case inl
      α : Type u_1
      n : Nat
      inst✝ : DecidableEq α
      s : Sym (Option α) n
      ⊢ Eq (SymOptionSuccEquiv.encode (SymOptionSuccEquiv.decode (Sum.inl s))) (Sum. …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      n : Nat
      inst✝ : DecidableEq α
      s : Sym α n.succ
      ⊢ Eq (SymOptionSuccEquiv.encode (SymOptionSuccEquiv.decode (Sum.inr s))) (Sum. …
    -/
  · unfold SymOptionSuccEquiv.encode
    /-
      case inr
      α : Type u_1
      n : Nat
      inst✝ : DecidableEq α
      s : Sym α n.succ
      ⊢ Eq (dite (Membership.mem (SymOptionSuccEquiv.decode (Sum.inr s)) Option.none …
    -/
    split_ifs with h
      /-
        case pos
        α : Type u_1
        n : Nat
        inst✝ : DecidableEq α
        s : Sym α n.succ
        h : Membership.mem (SymOptionSuccEquiv.decode (Sum.inr s)) Option.none
        ⊢ False
      -/
    · obtain ⟨a, _, ha⟩ := Multiset.mem_map.mp h
      /-
        case pos.intro.intro
        α : Type u_1
        n : Nat
        inst✝ : DecidableEq α
        s : Sym α n.succ
        h : Membership.mem (SymOptionSuccEquiv.decode (Sum.inr s)) Option.none
        a : α
        left✝ : Membership.mem (↑s) a
        ha : Eq (Function.Embedding.some a) Option.none
        ⊢ False
      -/
      exact Option.some_ne_none _ ha
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        n : Nat
        inst✝ : DecidableEq α
        s : Sym α n.succ
        h : Not (Membership.mem (SymOptionSuccEquiv.decode (Sum.inr s)) Option.none)
        ⊢ Eq (Sum.inr (Sym.map (fun o => (↑o).get ⋯) (SymOptionSuccEquiv.decode (Sum.i …
      -/
    · refine congr_arg Sum.inr ?_
      /-
        case neg
        α : Type u_1
        n : Nat
        inst✝ : DecidableEq α
        s : Sym α n.succ
        h : Not (Membership.mem (SymOptionSuccEquiv.decode (Sum.inr s)) Option.none)
        ⊢ Eq (Sym.map (fun o => (↑o).get ⋯) (SymOptionSuccEquiv.decode (Sum.inr s)).at …
      -/
      refine map_injective (Option.some_injective _) _ ?_
      /-
        case neg
        α : Type u_1
        n : Nat
        inst✝ : DecidableEq α
        s : Sym α n.succ
        h : Not (Membership.mem (SymOptionSuccEquiv.decode (Sum.inr s)) Option.none)
        ⊢ Eq (Sym.map Option.some (Sym.map (fun o => (↑o).get ⋯) (SymOptionSuccEquiv.d …
      -/
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/
      refine Eq.trans ?_ (.trans (SymOptionSuccEquiv.decode (Sum.inr s)).attach_map_coe ?_) <;> simp
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/


/-- The symmetric product over `Option` is a disjoint union over simpler symmetric products. -/
--@[simps]
def symOptionSuccEquiv [DecidableEq α] :
    Sym (Option α) n.succ ≃ Sym (Option α) n ⊕ Sym α n.succ where
  toFun := SymOptionSuccEquiv.encode
  invFun := SymOptionSuccEquiv.decode
  left_inv := SymOptionSuccEquiv.decode_encode
  right_inv := SymOptionSuccEquiv.encode_decode


