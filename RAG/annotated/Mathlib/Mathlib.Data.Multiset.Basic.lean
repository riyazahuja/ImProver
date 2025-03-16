/-- `Multiset α` is the quotient of `List α` by list permutation. The result
  is a type of finite sets with duplicates allowed. -/
def Multiset.{u} (α : Type u) : Type u :=
  Quotient (List.isSetoid α)


/-- The quotient map from `List α` to `Multiset α`. -/
@[coe]
def ofList : List α → Multiset α :=
  Quot.mk _


instance : Coe (List α) (Multiset α) :=
  ⟨ofList⟩


@[simp]
theorem quot_mk_to_coe (l : List α) : @Eq (Multiset α) ⟦l⟧ l :=
  rfl


@[simp]
theorem quot_mk_to_coe' (l : List α) : @Eq (Multiset α) (Quot.mk (· ≈ ·) l) l :=
  rfl


@[simp]
theorem quot_mk_to_coe'' (l : List α) : @Eq (Multiset α) (Quot.mk Setoid.r l) l :=
  rfl


@[simp]
theorem lift_coe {α β : Type*} (x : List α) (f : List α → β)
    (h : ∀ a b : List α, a ≈ b → f a = f b) : Quotient.lift f h (x : Multiset α) = f x :=
  Quotient.lift_mk _ _ _


@[simp]
theorem coe_eq_coe {l₁ l₂ : List α} : (l₁ : Multiset α) = l₂ ↔ l₁ ~ l₂ :=
  Quotient.eq

-- Porting note: new instance;
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: move to better place

instance [DecidableEq α] (l₁ l₂ : List α) : Decidable (l₁ ≈ l₂) :=
  inferInstanceAs (Decidable (l₁ ~ l₂))


instance [DecidableEq α] (l₁ l₂ : List α) : Decidable (isSetoid α l₁ l₂) :=
  inferInstanceAs (Decidable (l₁ ~ l₂))

-- Porting note: `Quotient.recOnSubsingleton₂ s₁ s₂` was in parens which broke elaboration

instance decidableEq [DecidableEq α] : DecidableEq (Multiset α)
  | s₁, s₂ => Quotient.recOnSubsingleton₂ s₁ s₂ fun _ _ => decidable_of_iff' _ Quotient.eq_iff_equiv


/-- defines a size for a multiset by referring to the size of the underlying list -/
protected
def sizeOf [SizeOf α] (s : Multiset α) : ℕ :=
  (Quot.liftOn s SizeOf.sizeOf) fun _ _ => Perm.sizeOf_eq_sizeOf


instance [SizeOf α] : SizeOf (Multiset α) :=
  ⟨Multiset.sizeOf⟩


/-- `0 : Multiset α` is the empty set -/
protected def zero : Multiset α :=
  @nil α


instance : Zero (Multiset α) :=
  ⟨Multiset.zero⟩


instance : EmptyCollection (Multiset α) :=
  ⟨0⟩


instance inhabitedMultiset : Inhabited (Multiset α) :=
  ⟨0⟩


instance [IsEmpty α] : Unique (Multiset α) where
  default := 0
             /-
               α : Type u_1
               β : Type v
               γ : Type u_2
               inst✝ : IsEmpty α
               ⊢ ∀ (a : Multiset α), Eq a Inhabited.default
             -/
  uniq := by rintro ⟨_ | ⟨a, l⟩⟩; exacts [rfl, isEmptyElim a]
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem coe_nil : (@nil α : Multiset α) = 0 :=
  rfl


@[simp]
theorem empty_eq_zero : (∅ : Multiset α) = 0 :=
  rfl


@[simp]
theorem coe_eq_zero (l : List α) : (l : Multiset α) = 0 ↔ l = [] :=
  Iff.trans coe_eq_coe perm_nil


theorem coe_eq_zero_iff_isEmpty (l : List α) : (l : Multiset α) = 0 ↔ l.isEmpty :=
  Iff.trans (coe_eq_zero l) isEmpty_iff.symm


/-- `cons a s` is the multiset which contains `s` plus one more instance of `a`. -/
def cons (a : α) (s : Multiset α) : Multiset α :=
  Quot.liftOn s (fun l => (a :: l : Multiset α)) fun _ _ p => Quot.sound (p.cons a)


@[inherit_doc Multiset.cons]
infixr:67 " ::ₘ " => Multiset.cons


instance : Insert α (Multiset α) :=
  ⟨cons⟩


@[simp]
theorem insert_eq_cons (a : α) (s : Multiset α) : insert a s = a ::ₘ s :=
  rfl


@[simp]
theorem cons_coe (a : α) (l : List α) : (a ::ₘ l : Multiset α) = (a :: l : List α) :=
  rfl


@[simp]
theorem cons_inj_left {a b : α} (s : Multiset α) : a ::ₘ s = b ::ₘ s ↔ a = b :=
  ⟨Quot.inductionOn s fun l e =>
      have : [a] ++ l ~ [b] ++ l := Quotient.exact e
      singleton_perm_singleton.1 <| (perm_append_right_iff _).1 this,
    congr_arg (· ::ₘ _)⟩


@[simp]
theorem cons_inj_right (a : α) : ∀ {s t : Multiset α}, a ::ₘ s = a ::ₘ t ↔ s = t := by
  /-
    α : Type u_1
    a : α
    ⊢ ∀ {s t : Multiset α}, Iff (Eq (Multiset.cons a s) (Multiset.cons a t)) (Eq s …
  -/
  rintro ⟨l₁⟩ ⟨l₂⟩; simp
                    /-
                      🎉 no goals
                    -/


@[elab_as_elim]
protected theorem induction {p : Multiset α → Prop} (empty : p 0)
    (cons : ∀ (a : α) (s : Multiset α), p s → p (a ::ₘ s)) : ∀ s, p s := by
  /-
    α : Type u_1
    p : Multiset α → Prop
    empty : p 0
    cons : ∀ (a : α) (s : Multiset α), p s → p (Multiset.cons a s)
    ⊢ ∀ (s : Multiset α), p s
  -/
              /-
                🎉 no goals
              -/
  rintro ⟨l⟩; induction l with | nil => exact empty | cons _ _ ih => exact cons _ _ ih


@[elab_as_elim]
protected theorem induction_on {p : Multiset α → Prop} (s : Multiset α) (empty : p 0)
    (cons : ∀ (a : α) (s : Multiset α), p s → p (a ::ₘ s)) : p s :=
  Multiset.induction empty cons s


theorem cons_swap (a b : α) (s : Multiset α) : a ::ₘ b ::ₘ s = b ::ₘ a ::ₘ s :=
  Quot.inductionOn s fun _ => Quotient.sound <| Perm.swap _ _ _


/-- Dependent recursor on multisets.
TODO: should be @[recursor 6], but then the definition of `Multiset.pi` fails with a stack
overflow in `whnf`.
-/
protected
def rec (C_0 : C 0) (C_cons : ∀ a m, C m → C (a ::ₘ m))
    (C_cons_heq :
      ∀ a a' m b, HEq (C_cons a (a' ::ₘ m) (C_cons a' m b)) (C_cons a' (a ::ₘ m) (C_cons a m b)))
    (m : Multiset α) : C m :=
  Quotient.hrecOn m (@List.rec α (fun l => C ⟦l⟧) C_0 fun a l b => C_cons a ⟦l⟧ b) fun _ _ h =>
    h.rec_heq
                     /-
                       α : Type u_1
                       β : Type v
                       γ : Type u_2
                       C : Multiset α → Sort u_3
                       C_0 : C 0
                       C_cons : (a : α) → (m : Multiset α) → C m → C (Multiset.cons a m)
                       C_cons_heq : ∀ (a a' : α) (m : Multiset α) (b : C m), HEq (C_cons a (Multiset. …
                       m : Multiset α
                       x✝² x✝¹ : List α
                       h : HasEquiv.Equiv x✝² x✝¹
                       a✝ : α
                       l✝ l'✝ : List α
                       b✝ : C (Quotient.mk (List.isSetoid α) l✝)
                       b'✝ : C (Quotient.mk (List.isSetoid α) l'✝)
                       hl : l✝.Perm l'✝
                       x✝ : HEq b✝ b'✝
                       ⊢ HEq (C_cons a✝ (Quotient.mk (List.isSetoid α) l✝) b✝) (C_cons a✝ (Quotient.m …
                     -/
      (fun hl _ ↦ by congr 1; exact Quot.sound hl)
                              /-
                                🎉 no goals
                              -/
      (C_cons_heq _ _ ⟦_⟧ _)


/-- Companion to `Multiset.rec` with more convenient argument order. -/
@[elab_as_elim]
protected
def recOn (m : Multiset α) (C_0 : C 0) (C_cons : ∀ a m, C m → C (a ::ₘ m))
    (C_cons_heq :
      ∀ a a' m b, HEq (C_cons a (a' ::ₘ m) (C_cons a' m b)) (C_cons a' (a ::ₘ m) (C_cons a m b))) :
    C m :=
  Multiset.rec C_0 C_cons C_cons_heq m


@[simp]
theorem recOn_0 : @Multiset.recOn α C (0 : Multiset α) C_0 C_cons C_cons_heq = C_0 :=
  rfl


@[simp]
theorem recOn_cons (a : α) (m : Multiset α) :
    (a ::ₘ m).recOn C_0 C_cons C_cons_heq = C_cons a m (m.recOn C_0 C_cons C_cons_heq) :=
  Quotient.inductionOn m fun _ => rfl


/-- `a ∈ s` means that `a` has nonzero multiplicity in `s`. -/
def Mem (s : Multiset α) (a : α) : Prop :=
  Quot.liftOn s (fun l => a ∈ l) fun l₁ l₂ (e : l₁ ~ l₂) => propext <| e.mem_iff


instance : Membership α (Multiset α) :=
  ⟨Mem⟩


@[simp]
theorem mem_coe {a : α} {l : List α} : a ∈ (l : Multiset α) ↔ a ∈ l :=
  Iff.rfl


instance decidableMem [DecidableEq α] (a : α) (s : Multiset α) : Decidable (a ∈ s) :=
  Quot.recOnSubsingleton s fun l ↦ inferInstanceAs (Decidable (a ∈ l))


@[simp]
theorem mem_cons {a b : α} {s : Multiset α} : a ∈ b ::ₘ s ↔ a = b ∨ a ∈ s :=
  Quot.inductionOn s fun _ => List.mem_cons


theorem mem_cons_of_mem {a b : α} {s : Multiset α} (h : a ∈ s) : a ∈ b ::ₘ s :=
  mem_cons.2 <| Or.inr h


theorem mem_cons_self (a : α) (s : Multiset α) : a ∈ a ::ₘ s :=
  mem_cons.2 (Or.inl rfl)


theorem forall_mem_cons {p : α → Prop} {a : α} {s : Multiset α} :
    (∀ x ∈ a ::ₘ s, p x) ↔ p a ∧ ∀ x ∈ s, p x :=
  Quotient.inductionOn' s fun _ => List.forall_mem_cons


theorem exists_cons_of_mem {s : Multiset α} {a : α} : a ∈ s → ∃ t, s = a ::ₘ t :=
  Quot.inductionOn s fun l (h : a ∈ l) =>
    let ⟨l₁, l₂, e⟩ := append_of_mem h
    e.symm ▸ ⟨(l₁ ++ l₂ : List α), Quot.sound perm_middle⟩


@[simp]
theorem not_mem_zero (a : α) : a ∉ (0 : Multiset α) :=
  List.not_mem_nil _


theorem eq_zero_of_forall_not_mem {s : Multiset α} : (∀ x, x ∉ s) → s = 0 :=
                                   /-
                                     α : Type u_1
                                     s : Multiset α
                                     l : List α
                                     H : ∀ (x : α), Not (Membership.mem (Quot.mk (⇑(List.isSetoid α)) l) x)
                                     ⊢ Eq (Quot.mk (⇑(List.isSetoid α)) l) 0
                                   -/
  Quot.inductionOn s fun l H => by rw [eq_nil_iff_forall_not_mem.mpr H]; rfl
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


theorem eq_zero_iff_forall_not_mem {s : Multiset α} : s = 0 ↔ ∀ a, a ∉ s :=
  ⟨fun h => h.symm ▸ fun _ => not_mem_zero _, eq_zero_of_forall_not_mem⟩


theorem exists_mem_of_ne_zero {s : Multiset α} : s ≠ 0 → ∃ a : α, a ∈ s :=
  Quot.inductionOn s fun l hl =>
    match l, hl with
    | [], h => False.elim <| h rfl
                          /-
                            α : Type u_1
                            s : Multiset α
                            l✝ : List α
                            hl : Ne (Quot.mk (⇑(List.isSetoid α)) l✝) 0
                            a : α
                            l : List α
                            x✝ : Ne (Quot.mk (⇑(List.isSetoid α)) (List.cons a l)) 0
                            ⊢ Membership.mem (Quot.mk (⇑(List.isSetoid α)) (List.cons a l)) a
                          -/
    | a :: l, _ => ⟨a, by simp⟩
                          /-
                            🎉 no goals
                          -/


theorem empty_or_exists_mem (s : Multiset α) : s = 0 ∨ ∃ a, a ∈ s :=
  or_iff_not_imp_left.mpr Multiset.exists_mem_of_ne_zero


@[simp]
theorem zero_ne_cons {a : α} {m : Multiset α} : 0 ≠ a ::ₘ m := fun h =>
  have : a ∈ (0 : Multiset α) := h.symm ▸ mem_cons_self _ _
  not_mem_zero _ this


@[simp]
theorem cons_ne_zero {a : α} {m : Multiset α} : a ::ₘ m ≠ 0 :=
  zero_ne_cons.symm


theorem cons_eq_cons {a b : α} {as bs : Multiset α} :
    a ::ₘ as = b ::ₘ bs ↔ a = b ∧ as = bs ∨ a ≠ b ∧ ∃ cs, as = b ::ₘ cs ∧ bs = a ::ₘ cs := by
  /-
    α : Type u_1
    a b : α
    as bs : Multiset α
    ⊢ Iff (Eq (Multiset.cons a as) (Multiset.cons b bs)) (Or (And (Eq a b) (Eq as  …
  -/
  haveI : DecidableEq α := Classical.decEq α
  /-
    α : Type u_1
    a b : α
    as bs : Multiset α
    this : DecidableEq α
    ⊢ Iff (Eq (Multiset.cons a as) (Multiset.cons b bs)) (Or (And (Eq a b) (Eq as  …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      a b : α
      as bs : Multiset α
      this : DecidableEq α
      ⊢ Eq (Multiset.cons a as) (Multiset.cons b bs) → Or (And (Eq a b) (Eq as bs))  …
    -/
  · intro eq
    /-
      case mp
      α : Type u_1
      a b : α
      as bs : Multiset α
      this : DecidableEq α
      eq : Eq (Multiset.cons a as) (Multiset.cons b bs)
      ⊢ Or (And (Eq a b) (Eq as bs)) (And (Ne a b) (Exists fun cs => And (Eq as (Mul …
    -/
    by_cases h : a = b
      /-
        case pos
        α : Type u_1
        a b : α
        as bs : Multiset α
        this : DecidableEq α
        eq : Eq (Multiset.cons a as) (Multiset.cons b bs)
        h : Eq a b
        ⊢ Or (And (Eq a b) (Eq as bs)) (And (Ne a b) (Exists fun cs => And (Eq as (Mul …
      -/
    · subst h
      /-
        case pos
        α : Type u_1
        a : α
        as bs : Multiset α
        this : DecidableEq α
        eq : Eq (Multiset.cons a as) (Multiset.cons a bs)
        ⊢ Or (And (Eq a a) (Eq as bs)) (And (Ne a a) (Exists fun cs => And (Eq as (Mul …
      -/
      simp_all
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        a b : α
        as bs : Multiset α
        this : DecidableEq α
        eq : Eq (Multiset.cons a as) (Multiset.cons b bs)
        h : Not (Eq a b)
        ⊢ Or (And (Eq a b) (Eq as bs)) (And (Ne a b) (Exists fun cs => And (Eq as (Mul …
      -/
    · have : a ∈ b ::ₘ bs := eq ▸ mem_cons_self _ _
      /-
        case neg
        α : Type u_1
        a b : α
        as bs : Multiset α
        this✝ : DecidableEq α
        eq : Eq (Multiset.cons a as) (Multiset.cons b bs)
        h : Not (Eq a b)
        this : Membership.mem (Multiset.cons b bs) a
        ⊢ Or (And (Eq a b) (Eq as bs)) (And (Ne a b) (Exists fun cs => And (Eq as (Mul …
      -/
      have : a ∈ bs := by simpa [h]
      /-
        case neg
        α : Type u_1
        a b : α
        as bs : Multiset α
        this✝¹ : DecidableEq α
        eq : Eq (Multiset.cons a as) (Multiset.cons b bs)
        h : Not (Eq a b)
        this✝ : Membership.mem (Multiset.cons b bs) a
        this : Membership.mem bs a
        ⊢ Or (And (Eq a b) (Eq as bs)) (And (Ne a b) (Exists fun cs => And (Eq as (Mul …
      -/
      rcases exists_cons_of_mem this with ⟨cs, hcs⟩
      simp only [h, hcs, false_and, ne_eq, not_false_eq_true, cons_inj_right, exists_eq_right',
        true_and, false_or]
      /-
        case neg.intro
        α : Type u_1
        a b : α
        as bs : Multiset α
        this✝¹ : DecidableEq α
        eq : Eq (Multiset.cons a as) (Multiset.cons b bs)
        h : Not (Eq a b)
        this✝ : Membership.mem (Multiset.cons b bs) a
        this : Membership.mem bs a
        cs : Multiset α
        hcs : Eq bs (Multiset.cons a cs)
        ⊢ Eq as (Multiset.cons b cs)
      -/
      have : a ::ₘ as = b ::ₘ a ::ₘ cs := by simp [eq, hcs]
      /-
        case neg.intro
        α : Type u_1
        a b : α
        as bs : Multiset α
        this✝² : DecidableEq α
        eq : Eq (Multiset.cons a as) (Multiset.cons b bs)
        h : Not (Eq a b)
        this✝¹ : Membership.mem (Multiset.cons b bs) a
        this✝ : Membership.mem bs a
        cs : Multiset α
        hcs : Eq bs (Multiset.cons a cs)
        this : Eq (Multiset.cons a as) (Multiset.cons b (Multiset.cons a cs))
        ⊢ Eq as (Multiset.cons b cs)
      -/
      have : a ::ₘ as = a ::ₘ b ::ₘ cs := by rwa [cons_swap]
      /-
        case neg.intro
        α : Type u_1
        a b : α
        as bs : Multiset α
        this✝³ : DecidableEq α
        eq : Eq (Multiset.cons a as) (Multiset.cons b bs)
        h : Not (Eq a b)
        this✝² : Membership.mem (Multiset.cons b bs) a
        this✝¹ : Membership.mem bs a
        cs : Multiset α
        hcs : Eq bs (Multiset.cons a cs)
        this✝ : Eq (Multiset.cons a as) (Multiset.cons b (Multiset.cons a cs))
        this : Eq (Multiset.cons a as) (Multiset.cons a (Multiset.cons b cs))
        ⊢ Eq as (Multiset.cons b cs)
      -/
      simpa using this
      /-
        🎉 no goals
      -/
    /-
      case mpr
      α : Type u_1
      a b : α
      as bs : Multiset α
      this : DecidableEq α
      ⊢ Or (And (Eq a b) (Eq as bs)) (And (Ne a b) (Exists fun cs => And (Eq as (Mul …
    -/
  · intro h
    /-
      case mpr
      α : Type u_1
      a b : α
      as bs : Multiset α
      this : DecidableEq α
      h : Or (And (Eq a b) (Eq as bs)) (And (Ne a b) (Exists fun cs => And (Eq as (M …
      ⊢ Eq (Multiset.cons a as) (Multiset.cons b bs)
    -/
    rcases h with (⟨eq₁, eq₂⟩ | ⟨_, cs, eq₁, eq₂⟩)
      /-
        case mpr.inl.intro
        α : Type u_1
        a b : α
        as bs : Multiset α
        this : DecidableEq α
        eq₁ : Eq a b
        eq₂ : Eq as bs
        ⊢ Eq (Multiset.cons a as) (Multiset.cons b bs)
      -/
    · simp [*]
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr.intro.intro.intro
        α : Type u_1
        a b : α
        as bs : Multiset α
        this : DecidableEq α
        left✝ : Ne a b
        cs : Multiset α
        eq₁ : Eq as (Multiset.cons b cs)
        eq₂ : Eq bs (Multiset.cons a cs)
        ⊢ Eq (Multiset.cons a as) (Multiset.cons b bs)
      -/
    · simp [*, cons_swap a b]
      /-
        🎉 no goals
      -/


instance : Singleton α (Multiset α) :=
  ⟨fun a => a ::ₘ 0⟩


instance : LawfulSingleton α (Multiset α) :=
  ⟨fun _ => rfl⟩


@[simp]
theorem cons_zero (a : α) : a ::ₘ 0 = {a} :=
  rfl


@[simp, norm_cast]
theorem coe_singleton (a : α) : ([a] : Multiset α) = {a} :=
  rfl


@[simp]
theorem mem_singleton {a b : α} : b ∈ ({a} : Multiset α) ↔ b = a := by
  /-
    α : Type u_1
    a b : α
    ⊢ Iff (Membership.mem (Singleton.singleton a) b) (Eq b a)
  -/
  simp only [← cons_zero, mem_cons, iff_self, or_false, not_mem_zero]
  /-
    🎉 no goals
  -/


theorem mem_singleton_self (a : α) : a ∈ ({a} : Multiset α) := by
  /-
    α : Type u_1
    a : α
    ⊢ Membership.mem (Singleton.singleton a) a
  -/
  rw [← cons_zero]
  /-
    α : Type u_1
    a : α
    ⊢ Membership.mem (Multiset.cons a 0) a
  -/
  exact mem_cons_self _ _
  /-
    🎉 no goals
  -/


@[simp]
theorem singleton_inj {a b : α} : ({a} : Multiset α) = {b} ↔ a = b := by
  /-
    α : Type u_1
    a b : α
    ⊢ Iff (Eq (Singleton.singleton a) (Singleton.singleton b)) (Eq a b)
  -/
  simp_rw [← cons_zero]
  /-
    α : Type u_1
    a b : α
    ⊢ Iff (Eq (Multiset.cons a 0) (Multiset.cons b 0)) (Eq a b)
  -/
  exact cons_inj_left _
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_eq_singleton {l : List α} {a : α} : (l : Multiset α) = {a} ↔ l = [a] := by
  /-
    α : Type u_1
    l : List α
    a : α
    ⊢ Iff (Eq (↑l) (Singleton.singleton a)) (Eq l (List.cons a List.nil))
  -/
  rw [← coe_singleton, coe_eq_coe, List.perm_singleton]
  /-
    🎉 no goals
  -/


@[simp]
theorem singleton_eq_cons_iff {a b : α} (m : Multiset α) : {a} = b ::ₘ m ↔ a = b ∧ m = 0 := by
  /-
    α : Type u_1
    a b : α
    m : Multiset α
    ⊢ Iff (Eq (Singleton.singleton a) (Multiset.cons b m)) (And (Eq a b) (Eq m 0))
  -/
  rw [← cons_zero, cons_eq_cons]
  /-
    α : Type u_1
    a b : α
    m : Multiset α
    ⊢ Iff (Or (And (Eq a b) (Eq 0 m)) (And (Ne a b) (Exists fun cs => And (Eq 0 (M …
  -/
  simp [eq_comm]
  /-
    🎉 no goals
  -/


theorem pair_comm (x y : α) : ({x, y} : Multiset α) = {y, x} :=
  cons_swap x y 0


/-- `s ⊆ t` is the lift of the list subset relation. It means that any
  element with nonzero multiplicity in `s` has nonzero multiplicity in `t`,
  but it does not imply that the multiplicity of `a` in `s` is less or equal than in `t`;
  see `s ≤ t` for this relation. -/
protected def Subset (s t : Multiset α) : Prop :=
  ∀ ⦃a : α⦄, a ∈ s → a ∈ t


instance : HasSubset (Multiset α) :=
  ⟨Multiset.Subset⟩


instance : HasSSubset (Multiset α) :=
  ⟨fun s t => s ⊆ t ∧ ¬t ⊆ s⟩


instance instIsNonstrictStrictOrder : IsNonstrictStrictOrder (Multiset α) (· ⊆ ·) (· ⊂ ·) where
  right_iff_left_not_left _ _ := Iff.rfl


@[simp]
theorem coe_subset {l₁ l₂ : List α} : (l₁ : Multiset α) ⊆ l₂ ↔ l₁ ⊆ l₂ :=
  Iff.rfl


@[simp]
theorem Subset.refl (s : Multiset α) : s ⊆ s := fun _ h => h


theorem Subset.trans {s t u : Multiset α} : s ⊆ t → t ⊆ u → s ⊆ u := fun h₁ h₂ _ m => h₂ (h₁ m)


theorem subset_iff {s t : Multiset α} : s ⊆ t ↔ ∀ ⦃x⦄, x ∈ s → x ∈ t :=
  Iff.rfl


theorem mem_of_subset {s t : Multiset α} {a : α} (h : s ⊆ t) : a ∈ s → a ∈ t :=
  @h _


@[simp]
theorem zero_subset (s : Multiset α) : 0 ⊆ s := fun a => (not_mem_nil a).elim


theorem subset_cons (s : Multiset α) (a : α) : s ⊆ a ::ₘ s := fun _ => mem_cons_of_mem


theorem ssubset_cons {s : Multiset α} {a : α} (ha : a ∉ s) : s ⊂ a ::ₘ s :=
  ⟨subset_cons _ _, fun h => ha <| h <| mem_cons_self _ _⟩


@[simp]
theorem cons_subset {a : α} {s t : Multiset α} : a ::ₘ s ⊆ t ↔ a ∈ t ∧ s ⊆ t := by
  /-
    α : Type u_1
    a : α
    s t : Multiset α
    ⊢ Iff (HasSubset.Subset (Multiset.cons a s) t) (And (Membership.mem t a) (HasS …
  -/
  simp [subset_iff, or_imp, forall_and]
  /-
    🎉 no goals
  -/


theorem cons_subset_cons {a : α} {s t : Multiset α} : s ⊆ t → a ::ₘ s ⊆ a ::ₘ t :=
  Quotient.inductionOn₂ s t fun _ _ => List.cons_subset_cons _


theorem eq_zero_of_subset_zero {s : Multiset α} (h : s ⊆ 0) : s = 0 :=
  eq_zero_of_forall_not_mem fun _ hx ↦ not_mem_zero _ (h hx)


@[simp] lemma subset_zero : s ⊆ 0 ↔ s = 0 :=
  ⟨eq_zero_of_subset_zero, fun xeq => xeq.symm ▸ Subset.refl 0⟩


                                                 /-
                                                   α : Type u_1
                                                   s : Multiset α
                                                   ⊢ Iff (HasSSubset.SSubset 0 s) (Ne s 0)
                                                 -/
@[simp] lemma zero_ssubset : 0 ⊂ s ↔ s ≠ 0 := by simp [ssubset_iff_subset_not_subset]
                                                 /-
                                                   🎉 no goals
                                                 -/


                                                       /-
                                                         α : Type u_1
                                                         s : Multiset α
                                                         a : α
                                                         ⊢ Iff (HasSubset.Subset (Singleton.singleton a) s) (Membership.mem s a)
                                                       -/
@[simp] lemma singleton_subset : {a} ⊆ s ↔ a ∈ s := by simp [subset_iff]
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem induction_on' {p : Multiset α → Prop} (S : Multiset α) (h₁ : p 0)
    (h₂ : ∀ {a s}, a ∈ S → s ⊆ S → p s → p (insert a s)) : p S :=
  @Multiset.induction_on α (fun T => T ⊆ S → p T) S (fun _ => h₁)
    (fun _ _ hps hs =>
      let ⟨hS, sS⟩ := cons_subset.1 hs
      h₂ hS sS (hps sS))
    (Subset.refl S)


/-- Produces a list of the elements in the multiset using choice. -/
noncomputable def toList (s : Multiset α) :=
  s.out


@[simp, norm_cast]
theorem coe_toList (s : Multiset α) : (s.toList : Multiset α) = s :=
  s.out_eq'


@[simp]
theorem toList_eq_nil {s : Multiset α} : s.toList = [] ↔ s = 0 := by
  /-
    α : Type u_1
    s : Multiset α
    ⊢ Iff (Eq s.toList List.nil) (Eq s 0)
  -/
  rw [← coe_eq_zero, coe_toList]
  /-
    🎉 no goals
  -/


                                                                       /-
                                                                         α : Type u_1
                                                                         s : Multiset α
                                                                         ⊢ Iff (Eq s.toList.isEmpty Bool.true) (Eq s 0)
                                                                       -/
theorem empty_toList {s : Multiset α} : s.toList.isEmpty ↔ s = 0 := by simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
theorem toList_zero : (Multiset.toList 0 : List α) = [] :=
  toList_eq_nil.mpr rfl


@[simp]
theorem mem_toList {a : α} {s : Multiset α} : a ∈ s.toList ↔ a ∈ s := by
  /-
    α : Type u_1
    a : α
    s : Multiset α
    ⊢ Iff (Membership.mem s.toList a) (Membership.mem s a)
  -/
  rw [← mem_coe, coe_toList]
  /-
    🎉 no goals
  -/


@[simp]
theorem toList_eq_singleton_iff {a : α} {m : Multiset α} : m.toList = [a] ↔ m = {a} := by
  /-
    α : Type u_1
    a : α
    m : Multiset α
    ⊢ Iff (Eq m.toList (List.cons a List.nil)) (Eq m (Singleton.singleton a))
  -/
  rw [← perm_singleton, ← coe_eq_coe, coe_toList, coe_singleton]
  /-
    🎉 no goals
  -/


@[simp]
theorem toList_singleton (a : α) : ({a} : Multiset α).toList = [a] :=
  Multiset.toList_eq_singleton_iff.2 rfl


/-- `s ≤ t` means that `s` is a sublist of `t` (up to permutation).
  Equivalently, `s ≤ t` means that `count a s ≤ count a t` for all `a`. -/
protected def Le (s t : Multiset α) : Prop :=
  (Quotient.liftOn₂ s t (· <+~ ·)) fun _ _ _ _ p₁ p₂ =>
    propext (p₂.subperm_left.trans p₁.subperm_right)


instance : PartialOrder (Multiset α) where
  le := Multiset.Le
                /-
                  α : Type u_1
                  β : Type v
                  γ : Type u_2
                  ⊢ ∀ (a : Multiset α), LE.le a a
                -/
  le_refl := by rintro ⟨l⟩; exact Subperm.refl _
                            /-
                              🎉 no goals
                            -/
                 /-
                   α : Type u_1
                   β : Type v
                   γ : Type u_2
                   ⊢ ∀ (a b c : Multiset α), LE.le a b → LE.le b c → LE.le a c
                 -/
  le_trans := by rintro ⟨l₁⟩ ⟨l₂⟩ ⟨l₃⟩; exact @Subperm.trans _ _ _ _
                                        /-
                                          🎉 no goals
                                        -/
                    /-
                      α : Type u_1
                      β : Type v
                      γ : Type u_2
                      ⊢ ∀ (a b : Multiset α), LE.le a b → LE.le b a → Eq a b
                    -/
  le_antisymm := by rintro ⟨l₁⟩ ⟨l₂⟩ h₁ h₂; exact Quot.sound (Subperm.antisymm h₁ h₂)
                                            /-
                                              🎉 no goals
                                            -/


instance decidableLE [DecidableEq α] : DecidableRel ((· ≤ ·) : Multiset α → Multiset α → Prop) :=
  fun s t => Quotient.recOnSubsingleton₂ s t List.decidableSubperm


theorem subset_of_le : s ≤ t → s ⊆ t :=
  Quotient.inductionOn₂ s t fun _ _ => Subperm.subset


alias Le.subset := subset_of_le


theorem mem_of_le (h : s ≤ t) : a ∈ s → a ∈ t :=
  mem_of_subset (subset_of_le h)


theorem not_mem_mono (h : s ⊆ t) : a ∉ t → a ∉ s :=
  mt <| @h _


@[simp]
theorem coe_le {l₁ l₂ : List α} : (l₁ : Multiset α) ≤ l₂ ↔ l₁ <+~ l₂ :=
  Iff.rfl


@[elab_as_elim]
theorem leInductionOn {C : Multiset α → Multiset α → Prop} {s t : Multiset α} (h : s ≤ t)
    (H : ∀ {l₁ l₂ : List α}, l₁ <+ l₂ → C l₁ l₂) : C s t :=
  Quotient.inductionOn₂ s t (fun l₁ _ ⟨l, p, s⟩ => (show ⟦l⟧ = ⟦l₁⟧ from Quot.sound p) ▸ H s) h


theorem zero_le (s : Multiset α) : 0 ≤ s :=
  Quot.inductionOn s fun l => (nil_sublist l).subperm


instance : OrderBot (Multiset α) where
  bot := 0
  bot_le := zero_le


/-- This is a `rfl` and `simp` version of `bot_eq_zero`. -/
@[simp]
theorem bot_eq_zero : (⊥ : Multiset α) = 0 :=
  rfl


theorem le_zero : s ≤ 0 ↔ s = 0 :=
  le_bot_iff


theorem lt_cons_self (s : Multiset α) (a : α) : s < a ::ₘ s :=
  Quot.inductionOn s fun l =>
                                           /-
                                             α : Type u_1
                                             s : Multiset α
                                             a : α
                                             l : List α
                                             this : And (l.Subperm (List.cons a l)) (Not (l.Perm (List.cons a l)))
                                             ⊢ LT.lt (Quot.mk (⇑(List.isSetoid α)) l) (Multiset.cons a (Quot.mk (⇑(List.isS …
                                           -/
    suffices l <+~ a :: l ∧ ¬l ~ a :: l by simpa [lt_iff_le_and_ne]
                                           /-
                                             🎉 no goals
                                           -/
    ⟨(sublist_cons_self _ _).subperm,
      fun p => _root_.ne_of_lt (lt_succ_self (length l)) p.length_eq⟩


theorem le_cons_self (s : Multiset α) (a : α) : s ≤ a ::ₘ s :=
  le_of_lt <| lt_cons_self _ _


theorem cons_le_cons_iff (a : α) : a ::ₘ s ≤ a ::ₘ t ↔ s ≤ t :=
  Quotient.inductionOn₂ s t fun _ _ => subperm_cons a


theorem cons_le_cons (a : α) : s ≤ t → a ::ₘ s ≤ a ::ₘ t :=
  (cons_le_cons_iff a).2


@[simp] lemma cons_lt_cons_iff : a ::ₘ s < a ::ₘ t ↔ s < t :=
  lt_iff_lt_of_le_iff_le' (cons_le_cons_iff _) (cons_le_cons_iff _)


lemma cons_lt_cons (a : α) (h : s < t) : a ::ₘ s < a ::ₘ t := cons_lt_cons_iff.2 h


theorem le_cons_of_not_mem (m : a ∉ s) : s ≤ a ::ₘ t ↔ s ≤ t := by
  /-
    α : Type u_1
    s t : Multiset α
    a : α
    m : Not (Membership.mem s a)
    ⊢ Iff (LE.le s (Multiset.cons a t)) (LE.le s t)
  -/
  refine ⟨?_, fun h => le_trans h <| le_cons_self _ _⟩
  suffices ∀ {t'}, s ≤ t' → a ∈ t' → a ::ₘ s ≤ t' by
    exact fun h => (cons_le_cons_iff a).1 (this h (mem_cons_self _ _))
  /-
    α : Type u_1
    s t : Multiset α
    a : α
    m : Not (Membership.mem s a)
    ⊢ ∀ {t' : Multiset α}, LE.le s t' → Membership.mem t' a → LE.le (Multiset.cons …
  -/
  introv h
  /-
    α : Type u_1
    s t : Multiset α
    a : α
    m : Not (Membership.mem s a)
    t' : Multiset α
    h : LE.le s t'
    ⊢ Membership.mem t' a → LE.le (Multiset.cons a s) t'
  -/
  revert m
  /-
    α : Type u_1
    s t : Multiset α
    a : α
    t' : Multiset α
    h : LE.le s t'
    ⊢ Not (Membership.mem s a) → Membership.mem t' a → LE.le (Multiset.cons a s) t'
  -/
  refine leInductionOn h ?_
  /-
    α : Type u_1
    s t : Multiset α
    a : α
    t' : Multiset α
    h : LE.le s t'
    ⊢ ∀ {l₁ l₂ : List α}, l₁.Sublist l₂ → Not (Membership.mem (↑l₁) a) → Membershi …
  -/
  introv s m₁ m₂
  /-
    α : Type u_1
    s✝ t : Multiset α
    a : α
    t' : Multiset α
    h : LE.le s✝ t'
    l₁ l₂ : List α
    s : l₁.Sublist l₂
    m₁ : Not (Membership.mem (↑l₁) a)
    m₂ : Membership.mem (↑l₂) a
    ⊢ LE.le (Multiset.cons a ↑l₁) ↑l₂
  -/
  rcases append_of_mem m₂ with ⟨r₁, r₂, rfl⟩
  exact
    perm_middle.subperm_left.2
      ((subperm_cons _).2 <| ((sublist_or_mem_of_sublist s).resolve_right m₁).subperm)


theorem cons_le_of_not_mem (hs : a ∉ s) : a ::ₘ s ≤ t ↔ a ∈ t ∧ s ≤ t := by
  /-
    α : Type u_1
    s t : Multiset α
    a : α
    hs : Not (Membership.mem s a)
    ⊢ Iff (LE.le (Multiset.cons a s) t) (And (Membership.mem t a) (LE.le s t))
  -/
  apply Iff.intro (fun h ↦ ⟨subset_of_le h (mem_cons_self a s), le_trans (le_cons_self s a) h⟩)
  /-
    α : Type u_1
    s t : Multiset α
    a : α
    hs : Not (Membership.mem s a)
    ⊢ And (Membership.mem t a) (LE.le s t) → LE.le (Multiset.cons a s) t
  -/
  rintro ⟨h₁, h₂⟩; rcases exists_cons_of_mem h₁ with ⟨_, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    s : Multiset α
    a : α
    hs : Not (Membership.mem s a)
    w✝ : Multiset α
    h₁ : Membership.mem (Multiset.cons a w✝) a
    h₂ : LE.le s (Multiset.cons a w✝)
    ⊢ LE.le (Multiset.cons a s) (Multiset.cons a w✝)
  -/
  exact cons_le_cons _ ((le_cons_of_not_mem hs).mp h₂)
  /-
    🎉 no goals
  -/


@[simp]
theorem singleton_ne_zero (a : α) : ({a} : Multiset α) ≠ 0 :=
  ne_of_gt (lt_cons_self _ _)


@[simp]
theorem zero_ne_singleton (a : α) : 0 ≠ ({a} : Multiset α) := singleton_ne_zero _ |>.symm


@[simp]
theorem singleton_le {a : α} {s : Multiset α} : {a} ≤ s ↔ a ∈ s :=
  ⟨fun h => mem_of_le h (mem_singleton_self _), fun h =>
    let ⟨_t, e⟩ := exists_cons_of_mem h
    e.symm ▸ cons_le_cons _ (zero_le _)⟩


@[simp] lemma le_singleton : s ≤ {a} ↔ s = 0 ∨ s = {a} :=
  Quot.induction_on s fun l ↦ by simp only [cons_zero, ← coe_singleton, quot_mk_to_coe'', coe_le,
    coe_eq_zero, coe_eq_coe, perm_singleton, subperm_singleton_iff]


@[simp] lemma lt_singleton : s < {a} ↔ s = 0 := by
  simp only [lt_iff_le_and_ne, le_singleton, or_and_right, Ne, and_not_self, or_false,
    and_iff_left_iff_imp]
  /-
    α : Type u_1
    s : Multiset α
    a : α
    ⊢ Eq s 0 → Not (Eq s (Singleton.singleton a))
  -/
  rintro rfl
  /-
    α : Type u_1
    a : α
    ⊢ Not (Eq 0 (Singleton.singleton a))
  -/
  exact (singleton_ne_zero _).symm
  /-
    🎉 no goals
  -/


@[simp] lemma ssubset_singleton_iff : s ⊂ {a} ↔ s = 0 := by
  /-
    α : Type u_1
    s : Multiset α
    a : α
    ⊢ Iff (HasSSubset.SSubset s (Singleton.singleton a)) (Eq s 0)
  -/
  refine ⟨fun hs ↦ eq_zero_of_subset_zero fun b hb ↦ (hs.2 ?_).elim, ?_⟩
    /-
      case refine_1
      α : Type u_1
      s : Multiset α
      a : α
      hs : HasSSubset.SSubset s (Singleton.singleton a)
      b : α
      hb : Membership.mem s b
      ⊢ HasSubset.Subset (Singleton.singleton a) s
    -/
  · obtain rfl := mem_singleton.1 (hs.1 hb)
    /-
      case refine_1
      α : Type u_1
      s : Multiset α
      b : α
      hb : Membership.mem s b
      hs : HasSSubset.SSubset s (Singleton.singleton b)
      ⊢ HasSubset.Subset (Singleton.singleton b) s
    -/
    rwa [singleton_subset]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      s : Multiset α
      a : α
      ⊢ Eq s 0 → HasSSubset.SSubset s (Singleton.singleton a)
    -/
  · rintro rfl
    /-
      case refine_2
      α : Type u_1
      a : α
      ⊢ HasSSubset.SSubset 0 (Singleton.singleton a)
    -/
    simp
    /-
      🎉 no goals
    -/


/-- The sum of two multisets is the lift of the list append operation.
  This adds the multiplicities of each element,
  i.e. `count a (s + t) = count a s + count a t`. -/
protected def add (s₁ s₂ : Multiset α) : Multiset α :=
  (Quotient.liftOn₂ s₁ s₂ fun l₁ l₂ => ((l₁ ++ l₂ : List α) : Multiset α)) fun _ _ _ _ p₁ p₂ =>
    Quot.sound <| p₁.append p₂


instance : Add (Multiset α) :=
  ⟨Multiset.add⟩


@[simp]
theorem coe_add (s t : List α) : (s + t : Multiset α) = (s ++ t : List α) :=
  rfl


@[simp]
theorem singleton_add (a : α) (s : Multiset α) : {a} + s = a ::ₘ s :=
  rfl


private theorem add_le_add_iff_left' {s t u : Multiset α} : s + t ≤ s + u ↔ t ≤ u :=
  Quotient.inductionOn₃ s t u fun _ _ _ => subperm_append_left _


instance : AddLeftMono (Multiset α) :=
  ⟨fun _s _t _u => add_le_add_iff_left'.2⟩


instance : AddLeftReflectLE (Multiset α) :=
  ⟨fun _s _t _u => add_le_add_iff_left'.1⟩


instance instAddCommMonoid : AddCancelCommMonoid (Multiset α) where
  add_comm := fun s t => Quotient.inductionOn₂ s t fun _ _ => Quot.sound perm_append_comm
  add_assoc := fun s₁ s₂ s₃ =>
    Quotient.inductionOn₃ s₁ s₂ s₃ fun l₁ l₂ l₃ => congr_arg _ <| append_assoc l₁ l₂ l₃
  zero_add := fun s => Quot.inductionOn s fun _ => rfl
  add_zero := fun s => Quotient.inductionOn s fun l => congr_arg _ <| append_nil l
  add_left_cancel := fun _ _ _ h =>
    le_antisymm (Multiset.add_le_add_iff_left'.mp h.le) (Multiset.add_le_add_iff_left'.mp h.ge)
  nsmul := nsmulRec


                                                          /-
                                                            α : Type u_1
                                                            s t : Multiset α
                                                            ⊢ LE.le s (HAdd.hAdd s t)
                                                          -/
theorem le_add_right (s t : Multiset α) : s ≤ s + t := by simpa using add_le_add_left (zero_le t) s
                                                          /-
                                                            🎉 no goals
                                                          -/


                                                         /-
                                                           α : Type u_1
                                                           s t : Multiset α
                                                           ⊢ LE.le s (HAdd.hAdd t s)
                                                         -/
theorem le_add_left (s t : Multiset α) : s ≤ t + s := by simpa using add_le_add_right (zero_le t) s
                                                         /-
                                                           🎉 no goals
                                                         -/


lemma subset_add_left {s t : Multiset α} : s ⊆ s + t := subset_of_le <| le_add_right s t


lemma subset_add_right {s t : Multiset α} : s ⊆ t + s := subset_of_le <| le_add_left s t


theorem le_iff_exists_add {s t : Multiset α} : s ≤ t ↔ ∃ u, t = s + u :=
  ⟨fun h =>
    leInductionOn h fun s =>
      let ⟨l, p⟩ := s.exists_perm_append
      ⟨l, Quot.sound p⟩,
    fun ⟨_u, e⟩ => e.symm ▸ le_add_right _ _⟩


@[simp]
theorem cons_add (a : α) (s t : Multiset α) : a ::ₘ s + t = a ::ₘ (s + t) := by
  /-
    α : Type u_1
    a : α
    s t : Multiset α
    ⊢ Eq (HAdd.hAdd (Multiset.cons a s) t) (Multiset.cons a (HAdd.hAdd s t))
  -/
  rw [← singleton_add, ← singleton_add, add_assoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem add_cons (a : α) (s t : Multiset α) : s + a ::ₘ t = a ::ₘ (s + t) := by
  /-
    α : Type u_1
    a : α
    s t : Multiset α
    ⊢ Eq (HAdd.hAdd s (Multiset.cons a t)) (Multiset.cons a (HAdd.hAdd s t))
  -/
  rw [add_comm, cons_add, add_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_add {a : α} {s t : Multiset α} : a ∈ s + t ↔ a ∈ s ∨ a ∈ t :=
  Quotient.inductionOn₂ s t fun _l₁ _l₂ => mem_append


theorem mem_of_mem_nsmul {a : α} {s : Multiset α} {n : ℕ} (h : a ∈ n • s) : a ∈ s := by
  /-
    α : Type u_1
    a : α
    s : Multiset α
    n : Nat
    h : Membership.mem (HSMul.hSMul n s) a
    ⊢ Membership.mem s a
  -/
  induction' n with n ih
    /-
      case zero
      α : Type u_1
      a : α
      s : Multiset α
      h : Membership.mem (HSMul.hSMul 0 s) a
      ⊢ Membership.mem s a
    -/
  · rw [zero_nsmul] at h
    /-
      case zero
      α : Type u_1
      a : α
      s : Multiset α
      h : Membership.mem 0 a
      ⊢ Membership.mem s a
    -/
    exact absurd h (not_mem_zero _)
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u_1
      a : α
      s : Multiset α
      n : Nat
      ih : Membership.mem (HSMul.hSMul n s) a → Membership.mem s a
      h : Membership.mem (HSMul.hSMul (HAdd.hAdd n 1) s) a
      ⊢ Membership.mem s a
    -/
  · rw [succ_nsmul, mem_add] at h
    /-
      case succ
      α : Type u_1
      a : α
      s : Multiset α
      n : Nat
      ih : Membership.mem (HSMul.hSMul n s) a → Membership.mem s a
      h : Or (Membership.mem (HSMul.hSMul n s) a) (Membership.mem s a)
      ⊢ Membership.mem s a
    -/
    exact h.elim ih id
    /-
      🎉 no goals
    -/


@[simp]
theorem mem_nsmul {a : α} {s : Multiset α} {n : ℕ} : a ∈ n • s ↔ n ≠ 0 ∧ a ∈ s := by
  /-
    α : Type u_1
    a : α
    s : Multiset α
    n : Nat
    ⊢ Iff (Membership.mem (HSMul.hSMul n s) a) (And (Ne n 0) (Membership.mem s a))
  -/
  refine ⟨fun ha ↦ ⟨?_, mem_of_mem_nsmul ha⟩, fun h => ?_⟩
    /-
      case refine_1
      α : Type u_1
      a : α
      s : Multiset α
      n : Nat
      ha : Membership.mem (HSMul.hSMul n s) a
      ⊢ Ne n 0
    -/
  · rintro rfl
    /-
      case refine_1
      α : Type u_1
      a : α
      s : Multiset α
      ha : Membership.mem (HSMul.hSMul 0 s) a
      ⊢ False
    -/
    simp [zero_nsmul] at ha
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    a : α
    s : Multiset α
    n : Nat
    h : And (Ne n 0) (Membership.mem s a)
    ⊢ Membership.mem (HSMul.hSMul n s) a
  -/
  obtain ⟨n, rfl⟩ := exists_eq_succ_of_ne_zero h.1
  /-
    case refine_2.intro
    α : Type u_1
    a : α
    s : Multiset α
    n : Nat
    h : And (Ne n.succ 0) (Membership.mem s a)
    ⊢ Membership.mem (HSMul.hSMul n.succ s) a
  -/
  rw [succ_nsmul, mem_add]
  /-
    case refine_2.intro
    α : Type u_1
    a : α
    s : Multiset α
    n : Nat
    h : And (Ne n.succ 0) (Membership.mem s a)
    ⊢ Or (Membership.mem (HSMul.hSMul n s) a) (Membership.mem s a)
  -/
  exact Or.inr h.2
  /-
    🎉 no goals
  -/


lemma mem_nsmul_of_ne_zero {a : α} {s : Multiset α} {n : ℕ} (h0 : n ≠ 0) : a ∈ n • s ↔ a ∈ s := by
  /-
    α : Type u_1
    a : α
    s : Multiset α
    n : Nat
    h0 : Ne n 0
    ⊢ Iff (Membership.mem (HSMul.hSMul n s) a) (Membership.mem s a)
  -/
  simp [*]
  /-
    🎉 no goals
  -/


theorem nsmul_cons {s : Multiset α} (n : ℕ) (a : α) :
    n • (a ::ₘ s) = n • ({a} : Multiset α) + n • s := by
  /-
    α : Type u_1
    s : Multiset α
    n : Nat
    a : α
    ⊢ Eq (HSMul.hSMul n (Multiset.cons a s)) (HAdd.hAdd (HSMul.hSMul n (Singleton. …
  -/
  rw [← singleton_add, nsmul_add]
  /-
    🎉 no goals
  -/


/-- The cardinality of a multiset is the sum of the multiplicities
  of all its elements, or simply the length of the underlying list. -/
def card (s : Multiset α) : ℕ := Quot.liftOn s length fun _l₁ _l₂ => Perm.length_eq


@[simp]
theorem coe_card (l : List α) : card (l : Multiset α) = length l :=
  rfl


@[simp]
theorem length_toList (s : Multiset α) : s.toList.length = card s := by
  /-
    α : Type u_1
    s : Multiset α
    ⊢ Eq s.toList.length s.card
  -/
  rw [← coe_card, coe_toList]
  /-
    🎉 no goals
  -/


@[simp]
theorem card_zero : @card α 0 = 0 :=
  rfl


@[simp] lemma card_add (s t : Multiset α) : card (s + t) = card s + card t :=
  Quotient.inductionOn₂ s t length_append


/-- `Multiset.card` bundled as a monoid hom. -/
@[simps]
def cardHom : Multiset α →+ ℕ where
  toFun := card
  map_zero' := card_zero
  map_add' := card_add


@[simp]
lemma card_nsmul (s : Multiset α) (n : ℕ) : card (n • s) = n * card s := cardHom.map_nsmul ..


@[simp]
theorem card_cons (a : α) (s : Multiset α) : card (a ::ₘ s) = card s + 1 :=
  Quot.inductionOn s fun _l => rfl


@[simp]
theorem card_singleton (a : α) : card ({a} : Multiset α) = 1 := by
  /-
    α : Type u_1
    a : α
    ⊢ Eq (Singleton.singleton a).card 1
  -/
  simp only [← cons_zero, card_zero, eq_self_iff_true, zero_add, card_cons]
  /-
    🎉 no goals
  -/


theorem card_pair (a b : α) : card {a, b} = 2 := by
  /-
    α : Type u_1
    a b : α
    ⊢ Eq (Insert.insert a (Singleton.singleton b)).card 2
  -/
  rw [insert_eq_cons, card_cons, card_singleton]
  /-
    🎉 no goals
  -/


theorem card_eq_one {s : Multiset α} : card s = 1 ↔ ∃ a, s = {a} :=
  ⟨Quot.inductionOn s fun _l h => (List.length_eq_one.1 h).imp fun _a => congr_arg _,
    fun ⟨_a, e⟩ => e.symm ▸ rfl⟩


theorem card_le_card {s t : Multiset α} (h : s ≤ t) : card s ≤ card t :=
  leInductionOn h Sublist.length_le


@[mono]
theorem card_mono : Monotone (@card α) := fun _a _b => card_le_card


theorem eq_of_le_of_card_le {s t : Multiset α} (h : s ≤ t) : card t ≤ card s → s = t :=
  leInductionOn h fun s h₂ => congr_arg _ <| s.eq_of_length_le h₂


theorem card_lt_card {s t : Multiset α} (h : s < t) : card s < card t :=
  lt_of_not_ge fun h₂ => _root_.ne_of_lt h <| eq_of_le_of_card_le (le_of_lt h) h₂


lemma card_strictMono : StrictMono (card : Multiset α → ℕ) := fun _ _ ↦ card_lt_card


theorem lt_iff_cons_le {s t : Multiset α} : s < t ↔ ∃ a, a ::ₘ s ≤ t :=
  ⟨Quotient.inductionOn₂ s t fun _l₁ _l₂ h =>
      Subperm.exists_of_length_lt (le_of_lt h) (card_lt_card h),
    fun ⟨_a, h⟩ => lt_of_lt_of_le (lt_cons_self _ _) h⟩


@[simp]
theorem card_eq_zero {s : Multiset α} : card s = 0 ↔ s = 0 :=
                                                                             /-
                                                                               α : Type u_1
                                                                               s : Multiset α
                                                                               e : Eq s 0
                                                                               ⊢ Eq s.card 0
                                                                             -/
  ⟨fun h => (eq_of_le_of_card_le (zero_le _) (le_of_eq h)).symm, fun e => by simp [e]⟩
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem card_pos {s : Multiset α} : 0 < card s ↔ s ≠ 0 :=
  Nat.pos_iff_ne_zero.trans <| not_congr card_eq_zero


theorem card_pos_iff_exists_mem {s : Multiset α} : 0 < card s ↔ ∃ a, a ∈ s :=
  Quot.inductionOn s fun _l => length_pos_iff_exists_mem


theorem card_eq_two {s : Multiset α} : card s = 2 ↔ ∃ x y, s = {x, y} :=
  ⟨Quot.inductionOn s fun _l h =>
      (List.length_eq_two.mp h).imp fun _a => Exists.imp fun _b => congr_arg _,
    fun ⟨_a, _b, e⟩ => e.symm ▸ rfl⟩


theorem card_eq_three {s : Multiset α} : card s = 3 ↔ ∃ x y z, s = {x, y, z} :=
  ⟨Quot.inductionOn s fun _l h =>
      (List.length_eq_three.mp h).imp fun _a =>
        Exists.imp fun _b => Exists.imp fun _c => congr_arg _,
    fun ⟨_a, _b, _c, e⟩ => e.symm ▸ rfl⟩


/-- The strong induction principle for multisets. -/
@[elab_as_elim]
def strongInductionOn {p : Multiset α → Sort*} (s : Multiset α) (ih : ∀ s, (∀ t < s, p t) → p s) :
    p s :=
    (ih s) fun t _h =>
      strongInductionOn t ih
termination_by card s
/-
  α : Type u_1
  p : Multiset α → Sort u_3
  s : Multiset α
  ih : (s : Multiset α) → ((t : Multiset α) → LT.lt t s → p t) → p s
  t : Multiset α
  _h : LT.lt t s
  ⊢ LT.lt t.card s.card
-/
decreasing_by exact card_lt_card _h
/-
  🎉 no goals
-/


theorem strongInductionOn_eq {p : Multiset α → Sort*} (s : Multiset α) (H) :
    @strongInductionOn _ p s H = H s fun t _h => @strongInductionOn _ p t H := by
  /-
    α : Type u_1
    p : Multiset α → Sort u_3
    s : Multiset α
    H : (s : Multiset α) → ((t : Multiset α) → LT.lt t s → p t) → p s
    ⊢ Eq (s.strongInductionOn H) (H s fun t _h => t.strongInductionOn H)
  -/
  rw [strongInductionOn]
  /-
    🎉 no goals
  -/


@[elab_as_elim]
theorem case_strongInductionOn {p : Multiset α → Prop} (s : Multiset α) (h₀ : p 0)
    (h₁ : ∀ a s, (∀ t ≤ s, p t) → p (a ::ₘ s)) : p s :=
  Multiset.strongInductionOn s fun s =>
    Multiset.induction_on s (fun _ => h₀) fun _a _s _ ih =>
      (h₁ _ _) fun _t h => ih _ <| lt_of_le_of_lt h <| lt_cons_self _ _


/-- Suppose that, given that `p t` can be defined on all supersets of `s` of cardinality less than
`n`, one knows how to define `p s`. Then one can inductively define `p s` for all multisets `s` of
cardinality less than `n`, starting from multisets of card `n` and iterating. This
can be used either to define data, or to prove properties. -/
def strongDownwardInduction {p : Multiset α → Sort*} {n : ℕ}
    (H : ∀ t₁, (∀ {t₂ : Multiset α}, card t₂ ≤ n → t₁ < t₂ → p t₂) → card t₁ ≤ n → p t₁)
    (s : Multiset α) :
    card s ≤ n → p s :=
  H s fun {t} ht _h =>
    strongDownwardInduction H t ht
termination_by n - card s
/-
  α : Type u_1
  n : Nat
  s t : Multiset α
  ht : LE.le t.card n
  _h : LT.lt s t
  ⊢ LT.lt (HSub.hSub n t.card) (HSub.hSub n s.card)
-/
decreasing_by simp_wf; have := (card_lt_card _h); omega
/-
  🎉 no goals
-/
-- Porting note: reorderd universes


theorem strongDownwardInduction_eq {p : Multiset α → Sort*} {n : ℕ}
    (H : ∀ t₁, (∀ {t₂ : Multiset α}, card t₂ ≤ n → t₁ < t₂ → p t₂) → card t₁ ≤ n → p t₁)
    (s : Multiset α) :
    strongDownwardInduction H s = H s fun ht _hst => strongDownwardInduction H _ ht := by
  /-
    α : Type u_1
    p : Multiset α → Sort u_3
    n : Nat
    H : (t₁ : Multiset α) → ({t₂ : Multiset α} → LE.le t₂.card n → LT.lt t₁ t₂ → p …
    s : Multiset α
    ⊢ Eq (Multiset.strongDownwardInduction H s) (H s fun {t₂} ht _hst => Multiset. …
  -/
  rw [strongDownwardInduction]
  /-
    🎉 no goals
  -/


/-- Analogue of `strongDownwardInduction` with order of arguments swapped. -/
@[elab_as_elim]
def strongDownwardInductionOn {p : Multiset α → Sort*} {n : ℕ} :
    ∀ s : Multiset α,
      (∀ t₁, (∀ {t₂ : Multiset α}, card t₂ ≤ n → t₁ < t₂ → p t₂) → card t₁ ≤ n → p t₁) →
        card s ≤ n → p s :=
  fun s H => strongDownwardInduction H s


theorem strongDownwardInductionOn_eq {p : Multiset α → Sort*} (s : Multiset α) {n : ℕ}
    (H : ∀ t₁, (∀ {t₂ : Multiset α}, card t₂ ≤ n → t₁ < t₂ → p t₂) → card t₁ ≤ n → p t₁) :
    s.strongDownwardInductionOn H = H s fun {t} ht _h => t.strongDownwardInductionOn H ht := by
  /-
    α : Type u_1
    p : Multiset α → Sort u_3
    s : Multiset α
    n : Nat
    H : (t₁ : Multiset α) → ({t₂ : Multiset α} → LE.le t₂.card n → LT.lt t₁ t₂ → p …
    ⊢ Eq (fun a => s.strongDownwardInductionOn H a) (H s fun {t} ht _h => t.strong …
  -/
  dsimp only [strongDownwardInductionOn]
  /-
    α : Type u_1
    p : Multiset α → Sort u_3
    s : Multiset α
    n : Nat
    H : (t₁ : Multiset α) → ({t₂ : Multiset α} → LE.le t₂.card n → LT.lt t₁ t₂ → p …
    ⊢ Eq (fun a => Multiset.strongDownwardInduction H s a) (H s fun {t} ht _h => M …
  -/
  rw [strongDownwardInduction]
  /-
    🎉 no goals
  -/


/-- Another way of expressing `strongInductionOn`: the `(<)` relation is well-founded. -/
instance instWellFoundedLT : WellFoundedLT (Multiset α) :=
  ⟨Subrelation.wf Multiset.card_lt_card (measure Multiset.card).2⟩


/-- `replicate n a` is the multiset containing only `a` with multiplicity `n`. -/
def replicate (n : ℕ) (a : α) : Multiset α :=
  List.replicate n a


theorem coe_replicate (n : ℕ) (a : α) : (List.replicate n a : Multiset α) = replicate n a := rfl


@[simp] theorem replicate_zero (a : α) : replicate 0 a = 0 := rfl


@[simp] theorem replicate_succ (a : α) (n) : replicate (n + 1) a = a ::ₘ replicate n a := rfl


theorem replicate_add (m n : ℕ) (a : α) : replicate (m + n) a = replicate m a + replicate n a :=
  congr_arg _ <| List.replicate_add ..


/-- `Multiset.replicate` as an `AddMonoidHom`. -/
@[simps]
def replicateAddMonoidHom (a : α) : ℕ →+ Multiset α where
  toFun := fun n => replicate n a
  map_zero' := replicate_zero a
  map_add' := fun _ _ => replicate_add _ _ a


theorem replicate_one (a : α) : replicate 1 a = {a} := rfl


@[simp] theorem card_replicate (n) (a : α) : card (replicate n a) = n :=
  length_replicate n a


theorem mem_replicate {a b : α} {n : ℕ} : b ∈ replicate n a ↔ n ≠ 0 ∧ b = a :=
  List.mem_replicate


theorem eq_of_mem_replicate {a b : α} {n} : b ∈ replicate n a → b = a :=
  List.eq_of_mem_replicate


theorem eq_replicate_card {a : α} {s : Multiset α} : s = replicate (card s) a ↔ ∀ b ∈ s, b = a :=
  Quot.inductionOn s fun _l => coe_eq_coe.trans <| perm_replicate.trans eq_replicate_length


alias ⟨_, eq_replicate_of_mem⟩ := eq_replicate_card


theorem eq_replicate {a : α} {n} {s : Multiset α} :
    s = replicate n a ↔ card s = n ∧ ∀ b ∈ s, b = a :=
  ⟨fun h => h.symm ▸ ⟨card_replicate _ _, fun _b => eq_of_mem_replicate⟩,
    fun ⟨e, al⟩ => e ▸ eq_replicate_of_mem al⟩


theorem replicate_right_injective {n : ℕ} (hn : n ≠ 0) : Injective (@replicate α n) :=
  fun _ _ h => (eq_replicate.1 h).2 _ <| mem_replicate.2 ⟨hn, rfl⟩


@[simp] theorem replicate_right_inj {a b : α} {n : ℕ} (h : n ≠ 0) :
    replicate n a = replicate n b ↔ a = b :=
  (replicate_right_injective h).eq_iff


theorem replicate_left_injective (a : α) : Injective (replicate · a) :=
  -- Porting note: was `fun m n h => by rw [← (eq_replicate.1 h).1, card_replicate]`
  LeftInverse.injective (card_replicate · a)


theorem replicate_subset_singleton (n : ℕ) (a : α) : replicate n a ⊆ {a} :=
  List.replicate_subset_singleton n a


theorem replicate_le_coe {a : α} {n} {l : List α} : replicate n a ≤ l ↔ List.replicate n a <+ l :=
  ⟨fun ⟨_l', p, s⟩ => perm_replicate.1 p ▸ s, Sublist.subperm⟩


theorem nsmul_replicate {a : α} (n m : ℕ) : n • replicate m a = replicate (n * m) a :=
  ((replicateAddMonoidHom a).map_nsmul _ _).symm


theorem nsmul_singleton (a : α) (n) : n • ({a} : Multiset α) = replicate n a := by
  /-
    α : Type u_1
    a : α
    n : Nat
    ⊢ Eq (HSMul.hSMul n (Singleton.singleton a)) (Multiset.replicate n a)
  -/
  rw [← replicate_one, nsmul_replicate, mul_one]
  /-
    🎉 no goals
  -/


theorem replicate_le_replicate (a : α) {k n : ℕ} : replicate k a ≤ replicate n a ↔ k ≤ n :=
                   /-
                     α : Type u_1
                     a : α
                     k n : Nat
                     ⊢ Iff (LE.le (Multiset.replicate k a) (Multiset.replicate n a)) ((List.replica …
                   -/
  _root_.trans (by rw [← replicate_le_coe, coe_replicate]) (List.replicate_sublist_replicate a)
                   /-
                     🎉 no goals
                   -/


@[gcongr]
theorem replicate_mono (a : α) {k n : ℕ} (h : k ≤ n) : replicate k a ≤ replicate n a :=
  (replicate_le_replicate a).2 h


theorem le_replicate_iff {m : Multiset α} {a : α} {n : ℕ} :
    m ≤ replicate n a ↔ ∃ k ≤ n, m = replicate k a :=
  ⟨fun h => ⟨card m, (card_mono h).trans_eq (card_replicate _ _),
      eq_replicate_card.2 fun _ hb => eq_of_mem_replicate <| subset_of_le h hb⟩,
    fun ⟨_, hkn, hm⟩ => hm.symm ▸ (replicate_le_replicate _).2 hkn⟩


theorem lt_replicate_succ {m : Multiset α} {x : α} {n : ℕ} :
    m < replicate (n + 1) x ↔ m ≤ replicate n x := by
  /-
    α : Type u_1
    m : Multiset α
    x : α
    n : Nat
    ⊢ Iff (LT.lt m (Multiset.replicate (HAdd.hAdd n 1) x)) (LE.le m (Multiset.repl …
  -/
  rw [lt_iff_cons_le]
  /-
    α : Type u_1
    m : Multiset α
    x : α
    n : Nat
    ⊢ Iff (Exists fun a => LE.le (Multiset.cons a m) (Multiset.replicate (HAdd.hAd …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      m : Multiset α
      x : α
      n : Nat
      ⊢ (Exists fun a => LE.le (Multiset.cons a m) (Multiset.replicate (HAdd.hAdd n  …
    -/
  · rintro ⟨x', hx'⟩
    /-
      case mp.intro
      α : Type u_1
      m : Multiset α
      x : α
      n : Nat
      x' : α
      hx' : LE.le (Multiset.cons x' m) (Multiset.replicate (HAdd.hAdd n 1) x)
      ⊢ LE.le m (Multiset.replicate n x)
    -/
    have := eq_of_mem_replicate (mem_of_le hx' (mem_cons_self _ _))
    /-
      case mp.intro
      α : Type u_1
      m : Multiset α
      x : α
      n : Nat
      x' : α
      hx' : LE.le (Multiset.cons x' m) (Multiset.replicate (HAdd.hAdd n 1) x)
      this : Eq x' x
      ⊢ LE.le m (Multiset.replicate n x)
    -/
    rwa [this, replicate_succ, cons_le_cons_iff] at hx'
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      m : Multiset α
      x : α
      n : Nat
      ⊢ LE.le m (Multiset.replicate n x) → Exists fun a => LE.le (Multiset.cons a m) …
    -/
  · intro h
    /-
      case mpr
      α : Type u_1
      m : Multiset α
      x : α
      n : Nat
      h : LE.le m (Multiset.replicate n x)
      ⊢ Exists fun a => LE.le (Multiset.cons a m) (Multiset.replicate (HAdd.hAdd n 1 …
    -/
    rw [replicate_succ]
    /-
      case mpr
      α : Type u_1
      m : Multiset α
      x : α
      n : Nat
      h : LE.le m (Multiset.replicate n x)
      ⊢ Exists fun a => LE.le (Multiset.cons a m) (Multiset.cons x (Multiset.replica …
    -/
    exact ⟨x, cons_le_cons _ h⟩
    /-
      🎉 no goals
    -/


/-- `erase s a` is the multiset that subtracts 1 from the multiplicity of `a`. -/
def erase (s : Multiset α) (a : α) : Multiset α :=
  Quot.liftOn s (fun l => (l.erase a : Multiset α)) fun _l₁ _l₂ p => Quot.sound (p.erase a)


@[simp]
theorem coe_erase (l : List α) (a : α) : erase (l : Multiset α) a = l.erase a :=
  rfl


@[simp]
theorem erase_zero (a : α) : (0 : Multiset α).erase a = 0 :=
  rfl


@[simp]
theorem erase_cons_head (a : α) (s : Multiset α) : (a ::ₘ s).erase a = s :=
  Quot.inductionOn s fun l => congr_arg _ <| List.erase_cons_head a l


@[simp]
theorem erase_cons_tail {a b : α} (s : Multiset α) (h : b ≠ a) :
    (b ::ₘ s).erase a = b ::ₘ s.erase a :=
  Quot.inductionOn s fun _ => congr_arg _ <| List.erase_cons_tail (not_beq_of_ne h)


@[simp]
theorem erase_singleton (a : α) : ({a} : Multiset α).erase a = 0 :=
  erase_cons_head a 0


@[simp]
theorem erase_of_not_mem {a : α} {s : Multiset α} : a ∉ s → s.erase a = s :=
  Quot.inductionOn s fun _l h => congr_arg _ <| List.erase_of_not_mem h


@[simp]
theorem cons_erase {s : Multiset α} {a : α} : a ∈ s → a ::ₘ s.erase a = s :=
  Quot.inductionOn s fun _l h => Quot.sound (perm_cons_erase h).symm


theorem erase_cons_tail_of_mem (h : a ∈ s) :
    (b ::ₘ s).erase a = b ::ₘ s.erase a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Multiset α
    a b : α
    h : Membership.mem s a
    ⊢ Eq ((Multiset.cons b s).erase a) (Multiset.cons b (s.erase a))
  -/
  rcases eq_or_ne a b with rfl | hab
    /-
      case inl
      α : Type u_1
      inst✝ : DecidableEq α
      s : Multiset α
      a : α
      h : Membership.mem s a
      ⊢ Eq ((Multiset.cons a s).erase a) (Multiset.cons a (s.erase a))
    -/
  · simp [cons_erase h]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝ : DecidableEq α
      s : Multiset α
      a b : α
      h : Membership.mem s a
      hab : Ne a b
      ⊢ Eq ((Multiset.cons b s).erase a) (Multiset.cons b (s.erase a))
    -/
  · exact s.erase_cons_tail hab.symm
    /-
      🎉 no goals
    -/


theorem le_cons_erase (s : Multiset α) (a : α) : s ≤ a ::ₘ s.erase a :=
  if h : a ∈ s then le_of_eq (cons_erase h).symm
          /-
            α : Type u_1
            inst✝ : DecidableEq α
            s : Multiset α
            a : α
            h : Not (Membership.mem s a)
            ⊢ LE.le s (Multiset.cons a (s.erase a))
          -/
  else by rw [erase_of_not_mem h]; apply le_cons_self
                                   /-
                                     🎉 no goals
                                   -/


theorem add_singleton_eq_iff {s t : Multiset α} {a : α} : s + {a} = t ↔ a ∈ t ∧ s = t.erase a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    a : α
    ⊢ Iff (Eq (HAdd.hAdd s (Singleton.singleton a)) t) (And (Membership.mem t a) ( …
  -/
  rw [add_comm, singleton_add]; constructor
    /-
      case mp
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Multiset α
      a : α
      ⊢ Eq (Multiset.cons a s) t → And (Membership.mem t a) (Eq s (t.erase a))
    -/
  · rintro rfl
    /-
      case mp
      α : Type u_1
      inst✝ : DecidableEq α
      s : Multiset α
      a : α
      ⊢ And (Membership.mem (Multiset.cons a s) a) (Eq s ((Multiset.cons a s).erase  …
    -/
    exact ⟨s.mem_cons_self a, (s.erase_cons_head a).symm⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Multiset α
      a : α
      ⊢ And (Membership.mem t a) (Eq s (t.erase a)) → Eq (Multiset.cons a s) t
    -/
  · rintro ⟨h, rfl⟩
    /-
      case mpr.intro
      α : Type u_1
      inst✝ : DecidableEq α
      t : Multiset α
      a : α
      h : Membership.mem t a
      ⊢ Eq (Multiset.cons a (t.erase a)) t
    -/
    exact cons_erase h
    /-
      🎉 no goals
    -/


theorem erase_add_left_pos {a : α} {s : Multiset α} (t) : a ∈ s → (s + t).erase a = s.erase a + t :=
  Quotient.inductionOn₂ s t fun _l₁ l₂ h => congr_arg _ <| erase_append_left l₂ h


theorem erase_add_right_pos {a : α} (s) {t : Multiset α} (h : a ∈ t) :
                                          /-
                                            α : Type u_1
                                            inst✝ : DecidableEq α
                                            a : α
                                            s t : Multiset α
                                            h : Membership.mem t a
                                            ⊢ Eq ((HAdd.hAdd s t).erase a) (HAdd.hAdd s (t.erase a))
                                          -/
    (s + t).erase a = s + t.erase a := by rw [add_comm, erase_add_left_pos s h, add_comm]
                                          /-
                                            🎉 no goals
                                          -/


theorem erase_add_right_neg {a : α} {s : Multiset α} (t) :
    a ∉ s → (s + t).erase a = s + t.erase a :=
  Quotient.inductionOn₂ s t fun _l₁ l₂ h => congr_arg _ <| erase_append_right l₂ h


theorem erase_add_left_neg {a : α} (s) {t : Multiset α} (h : a ∉ t) :
                                          /-
                                            α : Type u_1
                                            inst✝ : DecidableEq α
                                            a : α
                                            s t : Multiset α
                                            h : Not (Membership.mem t a)
                                            ⊢ Eq ((HAdd.hAdd s t).erase a) (HAdd.hAdd (s.erase a) t)
                                          -/
    (s + t).erase a = s.erase a + t := by rw [add_comm, erase_add_right_neg s h, add_comm]
                                          /-
                                            🎉 no goals
                                          -/


theorem erase_le (a : α) (s : Multiset α) : s.erase a ≤ s :=
  Quot.inductionOn s fun l => (erase_sublist a l).subperm


@[simp]
theorem erase_lt {a : α} {s : Multiset α} : s.erase a < s ↔ a ∈ s :=
  ⟨fun h => not_imp_comm.1 erase_of_not_mem (ne_of_lt h), fun h => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      s : Multiset α
      h : Membership.mem s a
      ⊢ LT.lt (s.erase a) s
    -/
    simpa [h] using lt_cons_self (s.erase a) a⟩
    /-
      🎉 no goals
    -/


theorem erase_subset (a : α) (s : Multiset α) : s.erase a ⊆ s :=
  subset_of_le (erase_le a s)


theorem mem_erase_of_ne {a b : α} {s : Multiset α} (ab : a ≠ b) : a ∈ s.erase b ↔ a ∈ s :=
  Quot.inductionOn s fun _l => List.mem_erase_of_ne ab


theorem mem_of_mem_erase {a b : α} {s : Multiset α} : a ∈ s.erase b → a ∈ s :=
  mem_of_subset (erase_subset _ _)


theorem erase_comm (s : Multiset α) (a b : α) : (s.erase a).erase b = (s.erase b).erase a :=
  Quot.inductionOn s fun l => congr_arg _ <| l.erase_comm a b


instance : RightCommutative erase (α := α) := ⟨erase_comm⟩


@[gcongr]
theorem erase_le_erase {s t : Multiset α} (a : α) (h : s ≤ t) : s.erase a ≤ t.erase a :=
  leInductionOn h fun h => (h.erase _).subperm


theorem erase_le_iff_le_cons {s t : Multiset α} {a : α} : s.erase a ≤ t ↔ s ≤ a ::ₘ t :=
  ⟨fun h => le_trans (le_cons_erase _ _) (cons_le_cons _ h), fun h =>
                         /-
                           α : Type u_1
                           inst✝ : DecidableEq α
                           s t : Multiset α
                           a : α
                           h : LE.le s (Multiset.cons a t)
                           m : Membership.mem s a
                           ⊢ LE.le (s.erase a) t
                         -/
    if m : a ∈ s then by rw [← cons_erase m] at h; exact (cons_le_cons_iff _).1 h
                                                   /-
                                                     🎉 no goals
                                                   -/
    else le_trans (erase_le _ _) ((le_cons_of_not_mem m).1 h)⟩


@[simp]
theorem card_erase_of_mem {a : α} {s : Multiset α} : a ∈ s → card (s.erase a) = pred (card s) :=
  Quot.inductionOn s fun _l => length_erase_of_mem


@[simp]
theorem card_erase_add_one {a : α} {s : Multiset α} : a ∈ s → card (s.erase a) + 1 = card s :=
  Quot.inductionOn s fun _l => length_erase_add_one


theorem card_erase_lt_of_mem {a : α} {s : Multiset α} : a ∈ s → card (s.erase a) < card s :=
  fun h => card_lt_card (erase_lt.mpr h)


theorem card_erase_le {a : α} {s : Multiset α} : card (s.erase a) ≤ card s :=
  card_le_card (erase_le a s)


theorem card_erase_eq_ite {a : α} {s : Multiset α} :
    card (s.erase a) = if a ∈ s then pred (card s) else card s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s : Multiset α
    ⊢ Eq (s.erase a).card (ite (Membership.mem s a) s.card.pred s.card)
  -/
  by_cases h : a ∈ s
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      s : Multiset α
      h : Membership.mem s a
      ⊢ Eq (s.erase a).card (ite (Membership.mem s a) s.card.pred s.card)
    -/
  · rwa [card_erase_of_mem h, if_pos]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      s : Multiset α
      h : Not (Membership.mem s a)
      ⊢ Eq (s.erase a).card (ite (Membership.mem s a) s.card.pred s.card)
    -/
  · rwa [erase_of_not_mem h, if_neg]
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_reverse (l : List α) : (reverse l : Multiset α) = l :=
  Quot.sound <| reverse_perm _


/-- `map f s` is the lift of the list `map` operation. The multiplicity
  of `b` in `map f s` is the number of `a ∈ s` (counting multiplicity)
  such that `f a = b`. -/
def map (f : α → β) (s : Multiset α) : Multiset β :=
  Quot.liftOn s (fun l : List α => (l.map f : Multiset β)) fun _l₁ _l₂ p => Quot.sound (p.map f)


@[congr]
theorem map_congr {f g : α → β} {s t : Multiset α} :
    s = t → (∀ x ∈ t, f x = g x) → map f s = map g t := by
  /-
    α : Type u_1
    β : Type v
    f g : α → β
    s t : Multiset α
    ⊢ Eq s t → (∀ (x : α), Membership.mem t x → Eq (f x) (g x)) → Eq (Multiset.map …
  -/
  rintro rfl h
  /-
    α : Type u_1
    β : Type v
    f g : α → β
    s : Multiset α
    h : ∀ (x : α), Membership.mem s x → Eq (f x) (g x)
    ⊢ Eq (Multiset.map f s) (Multiset.map g s)
  -/
  induction s using Quot.inductionOn
  /-
    case h
    α : Type u_1
    β : Type v
    f g : α → β
    a✝ : List α
    h : ∀ (x : α), Membership.mem (Quot.mk (⇑(List.isSetoid α)) a✝) x → Eq (f x) ( …
    ⊢ Eq (Multiset.map f (Quot.mk (⇑(List.isSetoid α)) a✝)) (Multiset.map g (Quot. …
  -/
  exact congr_arg _ (List.map_congr_left h)
  /-
    🎉 no goals
  -/


theorem map_hcongr {β' : Type v} {m : Multiset α} {f : α → β} {f' : α → β'} (h : β = β')
    (hf : ∀ a ∈ m, HEq (f a) (f' a)) : HEq (map f m) (map f' m) := by
  /-
    α : Type u_1
    β β' : Type v
    m : Multiset α
    f : α → β
    f' : α → β'
    h : Eq β β'
    hf : ∀ (a : α), Membership.mem m a → HEq (f a) (f' a)
    ⊢ HEq (Multiset.map f m) (Multiset.map f' m)
  -/
  subst h; simp at hf
  /-
    α : Type u_1
    β : Type v
    m : Multiset α
    f f' : α → β
    hf : ∀ (a : α), Membership.mem m a → Eq (f a) (f' a)
    ⊢ HEq (Multiset.map f m) (Multiset.map f' m)
  -/
  simp [map_congr rfl hf]
  /-
    🎉 no goals
  -/


theorem forall_mem_map_iff {f : α → β} {p : β → Prop} {s : Multiset α} :
    (∀ y ∈ s.map f, p y) ↔ ∀ x ∈ s, p (f x) :=
  Quotient.inductionOn' s fun _L => List.forall_mem_map


@[simp, norm_cast] lemma map_coe (f : α → β) (l : List α) : map f l = l.map f := rfl


@[simp]
theorem map_zero (f : α → β) : map f 0 = 0 :=
  rfl


@[simp]
theorem map_cons (f : α → β) (a s) : map f (a ::ₘ s) = f a ::ₘ map f s :=
  Quot.inductionOn s fun _l => rfl


theorem map_comp_cons (f : α → β) (t) : map f ∘ cons t = cons (f t) ∘ map f := by
  /-
    α : Type u_1
    β : Type v
    f : α → β
    t : α
    ⊢ Eq (Function.comp (Multiset.map f) (Multiset.cons t)) (Function.comp (Multis …
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type v
    f : α → β
    t : α
    x✝ : Multiset α
    ⊢ Eq (Function.comp (Multiset.map f) (Multiset.cons t) x✝) (Function.comp (Mul …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem map_singleton (f : α → β) (a : α) : ({a} : Multiset α).map f = {f a} :=
  rfl


@[simp]
theorem map_replicate (f : α → β) (k : ℕ) (a : α) : (replicate k a).map f = replicate k (f a) := by
  /-
    α : Type u_1
    β : Type v
    f : α → β
    k : Nat
    a : α
    ⊢ Eq (Multiset.map f (Multiset.replicate k a)) (Multiset.replicate k (f a))
  -/
  simp only [← coe_replicate, map_coe, List.map_replicate]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_add (f : α → β) (s t) : map f (s + t) = map f s + map f t :=
  Quotient.inductionOn₂ s t fun _l₁ _l₂ => congr_arg _ <| map_append _ _ _


/-- If each element of `s : Multiset α` can be lifted to `β`, then `s` can be lifted to
`Multiset β`. -/
instance canLift (c) (p) [CanLift α β c p] :
    CanLift (Multiset α) (Multiset β) (map c) fun s => ∀ x ∈ s, p x where
  prf := by
    /-
      α : Type u_1
      β : Type v
      γ : Type u_2
      c : β → α
      p : α → Prop
      inst✝ : CanLift α β c p
      ⊢ ∀ (x : Multiset α), (∀ (x_1 : α), Membership.mem x x_1 → p x_1) → Exists fun …
    -/
    rintro ⟨l⟩ hl
    /-
      case mk
      α : Type u_1
      β : Type v
      γ : Type u_2
      c : β → α
      p : α → Prop
      inst✝ : CanLift α β c p
      x✝ : Multiset α
      l : List α
      hl : ∀ (x : α), Membership.mem (Quot.mk (⇑(List.isSetoid α)) l) x → p x
      ⊢ Exists fun y => Eq (Multiset.map c y) (Quot.mk (⇑(List.isSetoid α)) l)
    -/
    lift l to List β using hl
    /-
      case mk.intro
      α : Type u_1
      β : Type v
      γ : Type u_2
      c : β → α
      p : α → Prop
      inst✝ : CanLift α β c p
      x✝ : Multiset α
      l : List β
      ⊢ Exists fun y => Eq (Multiset.map c y) (Quot.mk (⇑(List.isSetoid α)) (List.ma …
    -/
    exact ⟨l, map_coe _ _⟩
    /-
      🎉 no goals
    -/


/-- `Multiset.map` as an `AddMonoidHom`. -/
def mapAddMonoidHom (f : α → β) : Multiset α →+ Multiset β where
  toFun := map f
  map_zero' := map_zero _
  map_add' := map_add _


@[simp]
theorem coe_mapAddMonoidHom (f : α → β) :
    (mapAddMonoidHom f : Multiset α → Multiset β) = map f :=
  rfl


theorem map_nsmul (f : α → β) (n : ℕ) (s) : map f (n • s) = n • map f s :=
  (mapAddMonoidHom f).map_nsmul _ _


@[simp]
theorem mem_map {f : α → β} {b : β} {s : Multiset α} : b ∈ map f s ↔ ∃ a, a ∈ s ∧ f a = b :=
  Quot.inductionOn s fun _l => List.mem_map


@[simp]
theorem card_map (f : α → β) (s) : card (map f s) = card s :=
  Quot.inductionOn s fun _l => length_map _ _


@[simp]
theorem map_eq_zero {s : Multiset α} {f : α → β} : s.map f = 0 ↔ s = 0 := by
  /-
    α : Type u_1
    β : Type v
    s : Multiset α
    f : α → β
    ⊢ Iff (Eq (Multiset.map f s) 0) (Eq s 0)
  -/
  rw [← Multiset.card_eq_zero, Multiset.card_map, Multiset.card_eq_zero]
  /-
    🎉 no goals
  -/


theorem mem_map_of_mem (f : α → β) {a : α} {s : Multiset α} (h : a ∈ s) : f a ∈ map f s :=
  mem_map.2 ⟨_, h, rfl⟩


theorem map_eq_singleton {f : α → β} {s : Multiset α} {b : β} :
    map f s = {b} ↔ ∃ a : α, s = {a} ∧ f a = b := by
  /-
    α : Type u_1
    β : Type v
    f : α → β
    s : Multiset α
    b : β
    ⊢ Iff (Eq (Multiset.map f s) (Singleton.singleton b)) (Exists fun a => And (Eq …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type v
      f : α → β
      s : Multiset α
      b : β
      ⊢ Eq (Multiset.map f s) (Singleton.singleton b) → Exists fun a => And (Eq s (S …
    -/
  · intro h
    /-
      case mp
      α : Type u_1
      β : Type v
      f : α → β
      s : Multiset α
      b : β
      h : Eq (Multiset.map f s) (Singleton.singleton b)
      ⊢ Exists fun a => And (Eq s (Singleton.singleton a)) (Eq (f a) b)
    -/
    obtain ⟨a, ha⟩ : ∃ a, s = {a} := by rw [← card_eq_one, ← card_map, h, card_singleton]
    /-
      case mp.intro
      α : Type u_1
      β : Type v
      f : α → β
      s : Multiset α
      b : β
      h : Eq (Multiset.map f s) (Singleton.singleton b)
      a : α
      ha : Eq s (Singleton.singleton a)
      ⊢ Exists fun a => And (Eq s (Singleton.singleton a)) (Eq (f a) b)
    -/
    refine ⟨a, ha, ?_⟩
    /-
      case mp.intro
      α : Type u_1
      β : Type v
      f : α → β
      s : Multiset α
      b : β
      h : Eq (Multiset.map f s) (Singleton.singleton b)
      a : α
      ha : Eq s (Singleton.singleton a)
      ⊢ Eq (f a) b
    -/
    rw [← mem_singleton, ← h, ha, map_singleton, mem_singleton]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      β : Type v
      f : α → β
      s : Multiset α
      b : β
      ⊢ (Exists fun a => And (Eq s (Singleton.singleton a)) (Eq (f a) b)) → Eq (Mult …
    -/
  · rintro ⟨a, rfl, rfl⟩
    /-
      case mpr.intro.intro
      α : Type u_1
      β : Type v
      f : α → β
      a : α
      ⊢ Eq (Multiset.map f (Singleton.singleton a)) (Singleton.singleton (f a))
    -/
    simp
    /-
      🎉 no goals
    -/


theorem map_eq_cons [DecidableEq α] (f : α → β) (s : Multiset α) (t : Multiset β) (b : β) :
    (∃ a ∈ s, f a = b ∧ (s.erase a).map f = t) ↔ s.map f = b ::ₘ t := by
  /-
    α : Type u_1
    β : Type v
    inst✝ : DecidableEq α
    f : α → β
    s : Multiset α
    t : Multiset β
    b : β
    ⊢ Iff (Exists fun a => And (Membership.mem s a) (And (Eq (f a) b) (Eq (Multise …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type v
      inst✝ : DecidableEq α
      f : α → β
      s : Multiset α
      t : Multiset β
      b : β
      ⊢ (Exists fun a => And (Membership.mem s a) (And (Eq (f a) b) (Eq (Multiset.ma …
    -/
  · rintro ⟨a, ha, rfl, rfl⟩
    /-
      case mp.intro.intro.intro
      α : Type u_1
      β : Type v
      inst✝ : DecidableEq α
      f : α → β
      s : Multiset α
      a : α
      ha : Membership.mem s a
      ⊢ Eq (Multiset.map f s) (Multiset.cons (f a) (Multiset.map f (s.erase a)))
    -/
    rw [← map_cons, Multiset.cons_erase ha]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      β : Type v
      inst✝ : DecidableEq α
      f : α → β
      s : Multiset α
      t : Multiset β
      b : β
      ⊢ Eq (Multiset.map f s) (Multiset.cons b t) → Exists fun a => And (Membership. …
    -/
  · intro h
    have : b ∈ s.map f := by
      rw [h]
      exact mem_cons_self _ _
    /-
      case mpr
      α : Type u_1
      β : Type v
      inst✝ : DecidableEq α
      f : α → β
      s : Multiset α
      t : Multiset β
      b : β
      h : Eq (Multiset.map f s) (Multiset.cons b t)
      this : Membership.mem (Multiset.map f s) b
      ⊢ Exists fun a => And (Membership.mem s a) (And (Eq (f a) b) (Eq (Multiset.map …
    -/
    obtain ⟨a, h1, rfl⟩ := mem_map.mp this
    /-
      case mpr.intro.intro
      α : Type u_1
      β : Type v
      inst✝ : DecidableEq α
      f : α → β
      s : Multiset α
      t : Multiset β
      a : α
      h1 : Membership.mem s a
      h : Eq (Multiset.map f s) (Multiset.cons (f a) t)
      this : Membership.mem (Multiset.map f s) (f a)
      ⊢ Exists fun a_1 => And (Membership.mem s a_1) (And (Eq (f a_1) (f a)) (Eq (Mu …
    -/
    obtain ⟨u, rfl⟩ := exists_cons_of_mem h1
    /-
      case mpr.intro.intro.intro
      α : Type u_1
      β : Type v
      inst✝ : DecidableEq α
      f : α → β
      t : Multiset β
      a : α
      u : Multiset α
      h1 : Membership.mem (Multiset.cons a u) a
      h : Eq (Multiset.map f (Multiset.cons a u)) (Multiset.cons (f a) t)
      this : Membership.mem (Multiset.map f (Multiset.cons a u)) (f a)
      ⊢ Exists fun a_1 => And (Membership.mem (Multiset.cons a u) a_1) (And (Eq (f a …
    -/
    rw [map_cons, cons_inj_right] at h
    /-
      case mpr.intro.intro.intro
      α : Type u_1
      β : Type v
      inst✝ : DecidableEq α
      f : α → β
      t : Multiset β
      a : α
      u : Multiset α
      h1 : Membership.mem (Multiset.cons a u) a
      h : Eq (Multiset.map f u) t
      this : Membership.mem (Multiset.map f (Multiset.cons a u)) (f a)
      ⊢ Exists fun a_1 => And (Membership.mem (Multiset.cons a u) a_1) (And (Eq (f a …
    -/
    refine ⟨a, mem_cons_self _ _, rfl, ?_⟩
    /-
      case mpr.intro.intro.intro
      α : Type u_1
      β : Type v
      inst✝ : DecidableEq α
      f : α → β
      t : Multiset β
      a : α
      u : Multiset α
      h1 : Membership.mem (Multiset.cons a u) a
      h : Eq (Multiset.map f u) t
      this : Membership.mem (Multiset.map f (Multiset.cons a u)) (f a)
      ⊢ Eq (Multiset.map f ((Multiset.cons a u).erase a)) t
    -/
    rw [Multiset.erase_cons_head, h]
    /-
      🎉 no goals
    -/

-- The simpNF linter says that the LHS can be simplified via `Multiset.mem_map`.
-- However this is a higher priority lemma.
-- https://github.com/leanprover/std4/issues/207

@[simp 1100, nolint simpNF]
theorem mem_map_of_injective {f : α → β} (H : Function.Injective f) {a : α} {s : Multiset α} :
    f a ∈ map f s ↔ a ∈ s :=
  Quot.inductionOn s fun _l => List.mem_map_of_injective H


@[simp]
theorem map_map (g : β → γ) (f : α → β) (s : Multiset α) : map g (map f s) = map (g ∘ f) s :=
  Quot.inductionOn s fun _l => congr_arg _ <| List.map_map _ _ _


theorem map_id (s : Multiset α) : map id s = s :=
  Quot.inductionOn s fun _l => congr_arg _ <| List.map_id _


@[simp]
theorem map_id' (s : Multiset α) : map (fun x => x) s = s :=
  map_id s

-- Porting note: was a `simp` lemma in mathlib3

theorem map_const (s : Multiset α) (b : β) : map (const α b) s = replicate (card s) b :=
  Quot.inductionOn s fun _ => congr_arg _ <| List.map_const' _ _

-- Porting note: was not a `simp` lemma in mathlib3 because `Function.const` was reducible

@[simp] theorem map_const' (s : Multiset α) (b : β) : map (fun _ ↦ b) s = replicate (card s) b :=
  map_const _ _


theorem eq_of_mem_map_const {b₁ b₂ : β} {l : List α} (h : b₁ ∈ map (Function.const α b₂) l) :
    b₁ = b₂ :=
                                                         /-
                                                           α : Type u_1
                                                           β : Type v
                                                           b₁ b₂ : β
                                                           l : List α
                                                           h : Membership.mem (Multiset.map (Function.const α b₂) ↑l) b₁
                                                           ⊢ Membership.mem (Multiset.replicate (↑l).card b₂) b₁
                                                         -/
  eq_of_mem_replicate (n := card (l : Multiset α)) <| by rwa [map_const] at h
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp, gcongr]
theorem map_le_map {f : α → β} {s t : Multiset α} (h : s ≤ t) : map f s ≤ map f t :=
  leInductionOn h fun h => (h.map f).subperm


@[simp, gcongr]
theorem map_lt_map {f : α → β} {s t : Multiset α} (h : s < t) : s.map f < t.map f := by
  /-
    α : Type u_1
    β : Type v
    f : α → β
    s t : Multiset α
    h : LT.lt s t
    ⊢ LT.lt (Multiset.map f s) (Multiset.map f t)
  -/
  refine (map_le_map h.le).lt_of_not_le fun H => h.ne <| eq_of_le_of_card_le h.le ?_
  /-
    α : Type u_1
    β : Type v
    f : α → β
    s t : Multiset α
    h : LT.lt s t
    H : LE.le (Multiset.map f t) (Multiset.map f s)
    ⊢ LE.le t.card s.card
  -/
  rw [← s.card_map f, ← t.card_map f]
  /-
    α : Type u_1
    β : Type v
    f : α → β
    s t : Multiset α
    h : LT.lt s t
    H : LE.le (Multiset.map f t) (Multiset.map f s)
    ⊢ LE.le (Multiset.map f t).card (Multiset.map f s).card
  -/
  exact card_le_card H
  /-
    🎉 no goals
  -/


theorem map_mono (f : α → β) : Monotone (map f) := fun _ _ => map_le_map


theorem map_strictMono (f : α → β) : StrictMono (map f) := fun _ _ => map_lt_map


@[simp, gcongr]
theorem map_subset_map {f : α → β} {s t : Multiset α} (H : s ⊆ t) : map f s ⊆ map f t := fun _b m =>
  let ⟨a, h, e⟩ := mem_map.1 m
  mem_map.2 ⟨a, H h, e⟩


theorem map_erase [DecidableEq α] [DecidableEq β] (f : α → β) (hf : Function.Injective f) (x : α)
    (s : Multiset α) : (s.erase x).map f = (s.map f).erase (f x) := by
  /-
    α : Type u_1
    β : Type v
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    f : α → β
    hf : Function.Injective f
    x : α
    s : Multiset α
    ⊢ Eq (Multiset.map f (s.erase x)) ((Multiset.map f s).erase (f x))
  -/
  induction' s using Multiset.induction_on with y s ih
    /-
      case empty
      α : Type u_1
      β : Type v
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      f : α → β
      hf : Function.Injective f
      x : α
      ⊢ Eq (Multiset.map f (Multiset.erase 0 x)) ((Multiset.map f 0).erase (f x))
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_1
    β : Type v
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    f : α → β
    hf : Function.Injective f
    x y : α
    s : Multiset α
    ih : Eq (Multiset.map f (s.erase x)) ((Multiset.map f s).erase (f x))
    ⊢ Eq (Multiset.map f ((Multiset.cons y s).erase x)) ((Multiset.map f (Multiset …
  -/
  by_cases hxy : y = x
    /-
      case pos
      α : Type u_1
      β : Type v
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      f : α → β
      hf : Function.Injective f
      x y : α
      s : Multiset α
      ih : Eq (Multiset.map f (s.erase x)) ((Multiset.map f s).erase (f x))
      hxy : Eq y x
      ⊢ Eq (Multiset.map f ((Multiset.cons y s).erase x)) ((Multiset.map f (Multiset …
    -/
  · cases hxy
    /-
      case pos.refl
      α : Type u_1
      β : Type v
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      f : α → β
      hf : Function.Injective f
      x : α
      s : Multiset α
      ih : Eq (Multiset.map f (s.erase x)) ((Multiset.map f s).erase (f x))
      ⊢ Eq (Multiset.map f ((Multiset.cons x s).erase x)) ((Multiset.map f (Multiset …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type v
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      f : α → β
      hf : Function.Injective f
      x y : α
      s : Multiset α
      ih : Eq (Multiset.map f (s.erase x)) ((Multiset.map f s).erase (f x))
      hxy : Not (Eq y x)
      ⊢ Eq (Multiset.map f ((Multiset.cons y s).erase x)) ((Multiset.map f (Multiset …
    -/
  · rw [s.erase_cons_tail hxy, map_cons, map_cons, (s.map f).erase_cons_tail (hf.ne hxy), ih]
    /-
      🎉 no goals
    -/


theorem map_erase_of_mem [DecidableEq α] [DecidableEq β] (f : α → β)
    (s : Multiset α) {x : α} (h : x ∈ s) : (s.erase x).map f = (s.map f).erase (f x) := by
  /-
    α : Type u_1
    β : Type v
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    f : α → β
    s : Multiset α
    x : α
    h : Membership.mem s x
    ⊢ Eq (Multiset.map f (s.erase x)) ((Multiset.map f s).erase (f x))
  -/
  induction' s using Multiset.induction_on with y s ih
    /-
      case empty
      α : Type u_1
      β : Type v
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      f : α → β
      x : α
      h : Membership.mem 0 x
      ⊢ Eq (Multiset.map f (Multiset.erase 0 x)) ((Multiset.map f 0).erase (f x))
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_1
    β : Type v
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    f : α → β
    x y : α
    s : Multiset α
    ih : Membership.mem s x → Eq (Multiset.map f (s.erase x)) ((Multiset.map f s). …
    h : Membership.mem (Multiset.cons y s) x
    ⊢ Eq (Multiset.map f ((Multiset.cons y s).erase x)) ((Multiset.map f (Multiset …
  -/
  rcases eq_or_ne y x with rfl | hxy
    /-
      case cons.inl
      α : Type u_1
      β : Type v
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      f : α → β
      y : α
      s : Multiset α
      ih : Membership.mem s y → Eq (Multiset.map f (s.erase y)) ((Multiset.map f s). …
      h : Membership.mem (Multiset.cons y s) y
      ⊢ Eq (Multiset.map f ((Multiset.cons y s).erase y)) ((Multiset.map f (Multiset …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case cons.inr
    α : Type u_1
    β : Type v
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    f : α → β
    x y : α
    s : Multiset α
    ih : Membership.mem s x → Eq (Multiset.map f (s.erase x)) ((Multiset.map f s). …
    h : Membership.mem (Multiset.cons y s) x
    hxy : Ne y x
    ⊢ Eq (Multiset.map f ((Multiset.cons y s).erase x)) ((Multiset.map f (Multiset …
  -/
  replace h : x ∈ s := by simpa [hxy.symm] using h
  /-
    case cons.inr
    α : Type u_1
    β : Type v
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    f : α → β
    x y : α
    s : Multiset α
    ih : Membership.mem s x → Eq (Multiset.map f (s.erase x)) ((Multiset.map f s). …
    hxy : Ne y x
    h : Membership.mem s x
    ⊢ Eq (Multiset.map f ((Multiset.cons y s).erase x)) ((Multiset.map f (Multiset …
  -/
  rw [s.erase_cons_tail hxy, map_cons, map_cons, ih h, erase_cons_tail_of_mem (mem_map_of_mem f h)]
  /-
    🎉 no goals
  -/


theorem map_surjective_of_surjective {f : α → β} (hf : Function.Surjective f) :
    Function.Surjective (map f) := by
  /-
    α : Type u_1
    β : Type v
    f : α → β
    hf : Function.Surjective f
    ⊢ Function.Surjective (Multiset.map f)
  -/
  intro s
  /-
    α : Type u_1
    β : Type v
    f : α → β
    hf : Function.Surjective f
    s : Multiset β
    ⊢ Exists fun a => Eq (Multiset.map f a) s
  -/
  induction' s using Multiset.induction_on with x s ih
    /-
      case empty
      α : Type u_1
      β : Type v
      f : α → β
      hf : Function.Surjective f
      ⊢ Exists fun a => Eq (Multiset.map f a) 0
    -/
  · exact ⟨0, map_zero _⟩
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      β : Type v
      f : α → β
      hf : Function.Surjective f
      x : β
      s : Multiset β
      ih : Exists fun a => Eq (Multiset.map f a) s
      ⊢ Exists fun a => Eq (Multiset.map f a) (Multiset.cons x s)
    -/
  · obtain ⟨y, rfl⟩ := hf x
    /-
      case cons.intro
      α : Type u_1
      β : Type v
      f : α → β
      hf : Function.Surjective f
      s : Multiset β
      ih : Exists fun a => Eq (Multiset.map f a) s
      y : α
      ⊢ Exists fun a => Eq (Multiset.map f a) (Multiset.cons (f y) s)
    -/
    obtain ⟨t, rfl⟩ := ih
    /-
      case cons.intro.intro
      α : Type u_1
      β : Type v
      f : α → β
      hf : Function.Surjective f
      y : α
      t : Multiset α
      ⊢ Exists fun a => Eq (Multiset.map f a) (Multiset.cons (f y) (Multiset.map f t))
    -/
    exact ⟨y ::ₘ t, map_cons _ _ _⟩
    /-
      🎉 no goals
    -/


/-- `foldl f H b s` is the lift of the list operation `foldl f b l`,
  which folds `f` over the multiset. It is well defined when `f` is right-commutative,
  that is, `f (f b a₁) a₂ = f (f b a₂) a₁`. -/
def foldl (f : β → α → β) [RightCommutative f] (b : β) (s : Multiset α) : β :=
  Quot.liftOn s (fun l => List.foldl f b l) fun _l₁ _l₂ p => p.foldl_eq b


@[simp]
theorem foldl_zero (b) : foldl f b 0 = b :=
  rfl


@[simp]
theorem foldl_cons (b a s) : foldl f b (a ::ₘ s) = foldl f (f b a) s :=
  Quot.inductionOn s fun _l => rfl


@[simp]
theorem foldl_add (b s t) : foldl f b (s + t) = foldl f (foldl f b s) t :=
  Quotient.inductionOn₂ s t fun _l₁ _l₂ => foldl_append _ _ _ _


/-- `foldr f H b s` is the lift of the list operation `foldr f b l`,
  which folds `f` over the multiset. It is well defined when `f` is left-commutative,
  that is, `f a₁ (f a₂ b) = f a₂ (f a₁ b)`. -/
def foldr (f : α → β → β) [LeftCommutative f] (b : β) (s : Multiset α) : β :=
  Quot.liftOn s (fun l => List.foldr f b l) fun _l₁ _l₂ p => p.foldr_eq b


@[simp]
theorem foldr_zero (b) : foldr f b 0 = b :=
  rfl


@[simp]
theorem foldr_cons (b a s) : foldr f b (a ::ₘ s) = f a (foldr f b s) :=
  Quot.inductionOn s fun _l => rfl


@[simp]
theorem foldr_singleton (b a) : foldr f b ({a} : Multiset α) = f a b :=
  rfl


@[simp]
theorem foldr_add (b s t) : foldr f b (s + t) = foldr f (foldr f b t) s :=
  Quotient.inductionOn₂ s t fun _l₁ _l₂ => foldr_append _ _ _ _


@[simp]
theorem coe_foldr (f : α → β → β) [LeftCommutative f] (b : β) (l : List α) :
    foldr f b l = l.foldr f b :=
  rfl


@[simp]
theorem coe_foldl (f : β → α → β) [RightCommutative f] (b : β) (l : List α) :
    foldl f b l = l.foldl f b :=
  rfl


theorem coe_foldr_swap (f : α → β → β) [LeftCommutative f] (b : β) (l : List α) :
    foldr f b l = l.foldl (fun x y => f y x) b :=
  (congr_arg (foldr f b) (coe_reverse l)).symm.trans <| foldr_reverse _ _ _


theorem foldr_swap (f : α → β → β) [LeftCommutative f] (b : β) (s : Multiset α) :
    foldr f b s = foldl (fun x y => f y x) b s :=
  Quot.inductionOn s fun _l => coe_foldr_swap _ _ _


theorem foldl_swap (f : β → α → β) [RightCommutative f] (b : β) (s : Multiset α) :
    foldl f b s = foldr (fun x y => f y x) b s :=
  (foldr_swap _ _ _).symm


theorem foldr_induction' (f : α → β → β) [LeftCommutative f] (x : β) (q : α → Prop)
    (p : β → Prop) (s : Multiset α) (hpqf : ∀ a b, q a → p b → p (f a b)) (px : p x)
    (q_s : ∀ a ∈ s, q a) : p (foldr f x s) := by
  induction s using Multiset.induction with
  | empty => simpa
  | cons a s ihs =>
    simp only [forall_mem_cons, foldr_cons] at q_s ⊢
    exact hpqf _ _ q_s.1 (ihs q_s.2)


theorem foldr_induction (f : α → α → α) [LeftCommutative f] (x : α) (p : α → Prop)
    (s : Multiset α) (p_f : ∀ a b, p a → p b → p (f a b)) (px : p x) (p_s : ∀ a ∈ s, p a) :
    p (foldr f x s) :=
  foldr_induction' f x p p s p_f px p_s


theorem foldl_induction' (f : β → α → β) [RightCommutative f] (x : β) (q : α → Prop)
    (p : β → Prop) (s : Multiset α) (hpqf : ∀ a b, q a → p b → p (f b a)) (px : p x)
    (q_s : ∀ a ∈ s, q a) : p (foldl f x s) := by
  /-
    α : Type u_1
    β : Type v
    f : β → α → β
    inst✝ : RightCommutative f
    x : β
    q : α → Prop
    p : β → Prop
    s : Multiset α
    hpqf : ∀ (a : α) (b : β), q a → p b → p (f b a)
    px : p x
    q_s : ∀ (a : α), Membership.mem s a → q a
    ⊢ p (Multiset.foldl f x s)
  -/
  rw [foldl_swap]
  /-
    α : Type u_1
    β : Type v
    f : β → α → β
    inst✝ : RightCommutative f
    x : β
    q : α → Prop
    p : β → Prop
    s : Multiset α
    hpqf : ∀ (a : α) (b : β), q a → p b → p (f b a)
    px : p x
    q_s : ∀ (a : α), Membership.mem s a → q a
    ⊢ p (Multiset.foldr (fun x y => f y x) x s)
  -/
  exact foldr_induction' (fun x y => f y x) x q p s hpqf px q_s
  /-
    🎉 no goals
  -/


theorem foldl_induction (f : α → α → α) [RightCommutative f] (x : α) (p : α → Prop)
    (s : Multiset α) (p_f : ∀ a b, p a → p b → p (f b a)) (px : p x) (p_s : ∀ a ∈ s, p a) :
    p (foldl f x s) :=
  foldl_induction' f x p p s p_f px p_s


/-- Lift of the list `pmap` operation. Map a partial function `f` over a multiset
  `s` whose elements are all in the domain of `f`. -/
nonrec def pmap {p : α → Prop} (f : ∀ a, p a → β) (s : Multiset α) : (∀ a ∈ s, p a) → Multiset β :=
  Quot.recOn s (fun l H => ↑(pmap f l H)) fun l₁ l₂ (pp : l₁ ~ l₂) =>
    funext fun H₂ : ∀ a ∈ l₂, p a =>
      have H₁ : ∀ a ∈ l₁, p a := fun a h => H₂ a (pp.subset h)
      have : ∀ {s₂ e H}, @Eq.ndrec (Multiset α) l₁ (fun s => (∀ a ∈ s, p a) → Multiset β)
          (fun _ => ↑(pmap f l₁ H₁)) s₂ e H = ↑(pmap f l₁ H₁) := by
        /-
          α : Type u_1
          β : Type v
          γ : Type u_2
          p : α → Prop
          f : (a : α) → p a → β
          s : Multiset α
          l₁ l₂ : List α
          pp : l₁.Perm l₂
          H₂ : ∀ (a : α), Membership.mem l₂ a → p a
          H₁ : ∀ (a : α), Membership.mem l₁ a → p a
          ⊢ ∀ {s₂ : Multiset α} {e : Eq (↑l₁) s₂} {H : ∀ (a : α), Membership.mem s₂ a →  …
        -/
        intro s₂ e _; subst e; rfl
                               /-
                                 🎉 no goals
                               -/
      this.trans <| Quot.sound <| pp.pmap f


@[simp]
theorem coe_pmap {p : α → Prop} (f : ∀ a, p a → β) (l : List α) (H : ∀ a ∈ l, p a) :
    pmap f l H = l.pmap f H :=
  rfl


@[simp]
theorem pmap_zero {p : α → Prop} (f : ∀ a, p a → β) (h : ∀ a ∈ (0 : Multiset α), p a) :
    pmap f 0 h = 0 :=
  rfl


@[simp]
theorem pmap_cons {p : α → Prop} (f : ∀ a, p a → β) (a : α) (m : Multiset α) :
    ∀ h : ∀ b ∈ a ::ₘ m, p b,
      pmap f (a ::ₘ m) h =
        f a (h a (mem_cons_self a m)) ::ₘ pmap f m fun a ha => h a <| mem_cons_of_mem ha :=
  Quotient.inductionOn m fun _l _h => rfl


/-- "Attach" a proof that `a ∈ s` to each element `a` in `s` to produce
  a multiset on `{x // x ∈ s}`. -/
def attach (s : Multiset α) : Multiset { x // x ∈ s } :=
  pmap Subtype.mk s fun _a => id


@[simp]
theorem coe_attach (l : List α) : @Eq (Multiset { x // x ∈ l }) (@attach α l) l.attach :=
  rfl


theorem sizeOf_lt_sizeOf_of_mem [SizeOf α] {x : α} {s : Multiset α} (hx : x ∈ s) :
    SizeOf.sizeOf x < SizeOf.sizeOf s := by
  /-
    α : Type u_1
    inst✝ : SizeOf α
    x : α
    s : Multiset α
    hx : Membership.mem s x
    ⊢ LT.lt (SizeOf.sizeOf x) (SizeOf.sizeOf s)
  -/
  induction' s using Quot.inductionOn with l a b
  /-
    case h
    α : Type u_1
    inst✝ : SizeOf α
    x : α
    l : List α
    hx : Membership.mem (Quot.mk (⇑(List.isSetoid α)) l) x
    ⊢ LT.lt (SizeOf.sizeOf x) (SizeOf.sizeOf (Quot.mk (⇑(List.isSetoid α)) l))
  -/
  exact List.sizeOf_lt_sizeOf_of_mem hx
  /-
    🎉 no goals
  -/


theorem pmap_eq_map (p : α → Prop) (f : α → β) (s : Multiset α) :
    ∀ H, @pmap _ _ p (fun a _ => f a) s H = map f s :=
  Quot.inductionOn s fun l H => congr_arg _ <| List.pmap_eq_map p f l H


theorem pmap_congr {p q : α → Prop} {f : ∀ a, p a → β} {g : ∀ a, q a → β} (s : Multiset α) :
    ∀ {H₁ H₂}, (∀ a ∈ s, ∀ (h₁ h₂), f a h₁ = g a h₂) → pmap f s H₁ = pmap g s H₂ :=
  @(Quot.inductionOn s (fun l _H₁ _H₂ h => congr_arg _ <| List.pmap_congr_left l h))


theorem map_pmap {p : α → Prop} (g : β → γ) (f : ∀ a, p a → β) (s) :
    ∀ H, map g (pmap f s H) = pmap (fun a h => g (f a h)) s H :=
  Quot.inductionOn s fun l H => congr_arg _ <| List.map_pmap g f l H


theorem pmap_eq_map_attach {p : α → Prop} (f : ∀ a, p a → β) (s) :
    ∀ H, pmap f s H = s.attach.map fun x => f x.1 (H _ x.2) :=
  Quot.inductionOn s fun l H => congr_arg _ <| List.pmap_eq_map_attach f l H

-- @[simp] -- Porting note: Left hand does not simplify

theorem attach_map_val' (s : Multiset α) (f : α → β) : (s.attach.map fun i => f i.val) = s.map f :=
  Quot.inductionOn s fun l => congr_arg _ <| List.attach_map_coe l f


@[simp]
theorem attach_map_val (s : Multiset α) : s.attach.map Subtype.val = s :=
  (attach_map_val' _ _).trans s.map_id


@[simp]
theorem mem_attach (s : Multiset α) : ∀ x, x ∈ s.attach :=
  Quot.inductionOn s fun _l => List.mem_attach _


@[simp]
theorem mem_pmap {p : α → Prop} {f : ∀ a, p a → β} {s H b} :
    b ∈ pmap f s H ↔ ∃ (a : _) (h : a ∈ s), f a (H a h) = b :=
  Quot.inductionOn s (fun _l _H => List.mem_pmap) H


@[simp]
theorem card_pmap {p : α → Prop} (f : ∀ a, p a → β) (s H) : card (pmap f s H) = card s :=
  Quot.inductionOn s (fun _l _H => length_pmap) H


@[simp]
theorem card_attach {m : Multiset α} : card (attach m) = card m :=
  card_pmap _ _ _


@[simp]
theorem attach_zero : (0 : Multiset α).attach = 0 :=
  rfl


theorem attach_cons (a : α) (m : Multiset α) :
    (a ::ₘ m).attach =
      ⟨a, mem_cons_self a m⟩ ::ₘ m.attach.map fun p => ⟨p.1, mem_cons_of_mem p.2⟩ :=
  Quotient.inductionOn m fun l =>
    congr_arg _ <|
      congr_arg (List.cons _) <| by
        /-
          α : Type u_1
          a : α
          m : Multiset α
          l : List α
          ⊢ Eq (List.pmap Subtype.mk l ⋯) (List.map (fun p => ⟨↑p, ⋯⟩) (List.pmap Subtyp …
        -/
        rw [List.map_pmap]; exact List.pmap_congr_left _ fun _ _ _ _ => Subtype.eq rfl
                            /-
                              🎉 no goals
                            -/


/-- If `p` is a decidable predicate,
so is the predicate that all elements of a multiset satisfy `p`. -/
protected def decidableForallMultiset {p : α → Prop} [∀ a, Decidable (p a)] :
    Decidable (∀ a ∈ m, p a) :=
                                                                              /-
                                                                                α : Type u_1
                                                                                β : Type v
                                                                                γ : Type u_2
                                                                                m : Multiset α
                                                                                p : α → Prop
                                                                                inst✝ : (a : α) → Decidable (p a)
                                                                                l : List α
                                                                                ⊢ Iff (∀ (a : α), Membership.mem l a → p a) (∀ (a : α), Membership.mem (Quotie …
                                                                              -/
  Quotient.recOnSubsingleton m fun l => decidable_of_iff (∀ a ∈ l, p a) <| by simp
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


instance decidableDforallMultiset {p : ∀ a ∈ m, Prop} [_hp : ∀ (a) (h : a ∈ m), Decidable (p a h)] :
    Decidable (∀ (a) (h : a ∈ m), p a h) :=
  @decidable_of_iff _ _
    (Iff.intro (fun h a ha => h ⟨a, ha⟩ (mem_attach _ _)) fun h ⟨_a, _ha⟩ _ => h _ _)
    (@Multiset.decidableForallMultiset _ m.attach (fun a => p a.1 a.2) _)


/-- decidable equality for functions whose domain is bounded by multisets -/
instance decidableEqPiMultiset {β : α → Type*} [∀ a, DecidableEq (β a)] :
    DecidableEq (∀ a ∈ m, β a) := fun f g =>
                                                          /-
                                                            α : Type u_1
                                                            β✝ : Type v
                                                            γ : Type u_2
                                                            m : Multiset α
                                                            β : α → Type u_3
                                                            inst✝ : (a : α) → DecidableEq (β a)
                                                            f g : (a : α) → Membership.mem m a → β a
                                                            ⊢ Iff (∀ (a : α) (h : Membership.mem m a), Eq (f a h) (g a h)) (Eq f g)
                                                          -/
  decidable_of_iff (∀ (a) (h : a ∈ m), f a h = g a h) (by simp [funext_iff])
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- If `p` is a decidable predicate,
so is the existence of an element in a multiset satisfying `p`. -/
protected def decidableExistsMultiset {p : α → Prop} [DecidablePred p] : Decidable (∃ x ∈ m, p x) :=
                                                                              /-
                                                                                α : Type u_1
                                                                                β : Type v
                                                                                γ : Type u_2
                                                                                m : Multiset α
                                                                                p : α → Prop
                                                                                inst✝ : DecidablePred p
                                                                                l : List α
                                                                                ⊢ Iff (Exists fun a => And (Membership.mem l a) (p a)) (Exists fun x => And (M …
                                                                              -/
  Quotient.recOnSubsingleton m fun l => decidable_of_iff (∃ a ∈ l, p a) <| by simp
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


instance decidableDexistsMultiset {p : ∀ a ∈ m, Prop} [_hp : ∀ (a) (h : a ∈ m), Decidable (p a h)] :
    Decidable (∃ (a : _) (h : a ∈ m), p a h) :=
  @decidable_of_iff _ _
    (Iff.intro (fun ⟨⟨a, ha₁⟩, _, ha₂⟩ => ⟨a, ha₁, ha₂⟩) fun ⟨a, ha₁, ha₂⟩ =>
      ⟨⟨a, ha₁⟩, mem_attach _ _, ha₂⟩)
    (@Multiset.decidableExistsMultiset { a // a ∈ m } m.attach (fun a => p a.1 a.2) _)


/-- `s - t` is the multiset such that `count a (s - t) = count a s - count a t` for all `a`
  (note that it is truncated subtraction, so it is `0` if `count a t ≥ count a s`). -/
protected def sub (s t : Multiset α) : Multiset α :=
  (Quotient.liftOn₂ s t fun l₁ l₂ => (l₁.diff l₂ : Multiset α)) fun _v₁ _v₂ _w₁ _w₂ p₁ p₂ =>
    Quot.sound <| p₁.diff p₂


instance : Sub (Multiset α) :=
  ⟨Multiset.sub⟩


@[simp]
theorem coe_sub (s t : List α) : (s - t : Multiset α) = (s.diff t : List α) :=
  rfl


/-- This is a special case of `tsub_zero`, which should be used instead of this.
  This is needed to prove `OrderedSub (Multiset α)`. -/
protected theorem sub_zero (s : Multiset α) : s - 0 = s :=
  Quot.inductionOn s fun _l => rfl


@[simp]
theorem sub_cons (a : α) (s t : Multiset α) : s - a ::ₘ t = s.erase a - t :=
  Quotient.inductionOn₂ s t fun _l₁ _l₂ => congr_arg _ <| diff_cons _ _ _


protected theorem zero_sub (t : Multiset α) : 0 - t = 0 :=
                                               /-
                                                 α : Type u_1
                                                 inst✝ : DecidableEq α
                                                 t : Multiset α
                                                 a : α
                                                 s : Multiset α
                                                 ih : Eq (HSub.hSub 0 s) 0
                                                 ⊢ Eq (HSub.hSub 0 (Multiset.cons a s)) 0
                                               -/
  Multiset.induction_on t rfl fun a s ih => by simp [ih]
                                               /-
                                                 🎉 no goals
                                               -/


/-- This is a special case of `tsub_le_iff_right`, which should be used instead of this.
  This is needed to prove `OrderedSub (Multiset α)`. -/
protected theorem sub_le_iff_le_add : s - t ≤ u ↔ s ≤ u + t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t u : Multiset α
    ⊢ Iff (LE.le (HSub.hSub s t) u) (LE.le s (HAdd.hAdd u t))
  -/
  revert s
  exact @(Multiset.induction_on t (by simp [Multiset.sub_zero]) fun a t IH s => by
      simp [IH, erase_le_iff_le_cons])


protected theorem sub_le_self (s t : Multiset α) : s - t ≤ s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    ⊢ LE.le (HSub.hSub s t) s
  -/
  rw [Multiset.sub_le_iff_le_add]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    ⊢ LE.le s (HAdd.hAdd s t)
  -/
  exact le_add_right _ _
  /-
    🎉 no goals
  -/


instance : OrderedSub (Multiset α) :=
  ⟨fun _n _m _k => Multiset.sub_le_iff_le_add⟩


instance : ExistsAddOfLE (Multiset α) where
  exists_add_of_le h := leInductionOn h fun s =>
      let ⟨l, p⟩ := s.exists_perm_append
      ⟨l, Quot.sound p⟩


theorem cons_sub_of_le (a : α) {s t : Multiset α} (h : t ≤ s) : a ::ₘ s - t = a ::ₘ (s - t) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s t : Multiset α
    h : LE.le t s
    ⊢ Eq (HSub.hSub (Multiset.cons a s) t) (Multiset.cons a (HSub.hSub s t))
  -/
  rw [← singleton_add, ← singleton_add, add_tsub_assoc_of_le h]
  /-
    🎉 no goals
  -/


theorem sub_eq_fold_erase (s t : Multiset α) : s - t = foldl erase s t :=
  Quotient.inductionOn₂ s t fun l₁ l₂ => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Multiset α
      l₁ l₂ : List α
      ⊢ Eq (HSub.hSub (Quotient.mk (List.isSetoid α) l₁) (Quotient.mk (List.isSetoid …
    -/
    show ofList (l₁.diff l₂) = foldl erase l₁ l₂
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Multiset α
      l₁ l₂ : List α
      ⊢ Eq (↑(l₁.diff l₂)) (Multiset.foldl Multiset.erase ↑l₁ ↑l₂)
    -/
    rw [diff_eq_foldl l₁ l₂]
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Multiset α
      l₁ l₂ : List α
      ⊢ Eq (↑(List.foldl List.erase l₁ l₂)) (Multiset.foldl Multiset.erase ↑l₁ ↑l₂)
    -/
    symm
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Multiset α
      l₁ l₂ : List α
      ⊢ Eq (Multiset.foldl Multiset.erase ↑l₁ ↑l₂) ↑(List.foldl List.erase l₁ l₂)
    -/
    exact foldl_hom _ _ _ _ _ fun x y => rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem card_sub {s t : Multiset α} (h : t ≤ s) : card (s - t) = card s - card t :=
                             /-
                               α : Type u_1
                               inst✝ : DecidableEq α
                               s t : Multiset α
                               h : LE.le t s
                               ⊢ Eq (HAdd.hAdd (HSub.hSub s t).card t.card) s.card
                             -/
  Nat.eq_sub_of_add_eq <| by rw [← card_add, tsub_add_cancel_of_le h]
                             /-
                               🎉 no goals
                             -/


/-- `s ∪ t` is the lattice join operation with respect to the
  multiset `≤`. The multiplicity of `a` in `s ∪ t` is the maximum
  of the multiplicities in `s` and `t`. -/
def union (s t : Multiset α) : Multiset α :=
  s - t + t


instance : Union (Multiset α) :=
  ⟨union⟩


theorem union_def (s t : Multiset α) : s ∪ t = s - t + t :=
  rfl


theorem le_union_left {s t : Multiset α} : s ≤ s ∪ t :=
  le_tsub_add


theorem le_union_right {s t : Multiset α} : t ≤ s ∪ t :=
  le_add_left _ _


theorem eq_union_left : t ≤ s → s ∪ t = s :=
  tsub_add_cancel_of_le


@[gcongr]
theorem union_le_union_right (h : s ≤ t) (u) : s ∪ u ≤ t ∪ u :=
  add_le_add_right (tsub_le_tsub_right h _) u


theorem union_le (h₁ : s ≤ u) (h₂ : t ≤ u) : s ∪ t ≤ u := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t u : Multiset α
    h₁ : LE.le s u
    h₂ : LE.le t u
    ⊢ LE.le (Union.union s t) u
  -/
  rw [← eq_union_left h₂]; exact union_le_union_right h₁ t
                           /-
                             🎉 no goals
                           -/



@[simp]
theorem mem_union : a ∈ s ∪ t ↔ a ∈ s ∨ a ∈ t :=
  ⟨fun h => (mem_add.1 h).imp_left (mem_of_le <| Multiset.sub_le_self _ _),
    (Or.elim · (mem_of_le le_union_left) (mem_of_le le_union_right))⟩


@[simp]
theorem map_union [DecidableEq β] {f : α → β} (finj : Function.Injective f) {s t : Multiset α} :
    map f (s ∪ t) = map f s ∪ map f t :=
  Quotient.inductionOn₂ s t fun l₁ l₂ =>
                         /-
                           α : Type u_1
                           β : Type v
                           inst✝¹ : DecidableEq α
                           inst✝ : DecidableEq β
                           f : α → β
                           finj : Function.Injective f
                           s t : Multiset α
                           l₁ l₂ : List α
                           ⊢ Eq (List.map f (HAppend.hAppend (l₁.diff l₂) l₂)) (HAppend.hAppend ((List.ma …
                         -/
    congr_arg ofList (by rw [List.map_append f, List.map_diff finj])
                         /-
                           🎉 no goals
                         -/


@[simp] theorem zero_union : 0 ∪ s = s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Multiset α
    ⊢ Eq (Union.union 0 s) s
  -/
  simp [union_def, Multiset.zero_sub]
  /-
    🎉 no goals
  -/


@[simp] theorem union_zero : s ∪ 0 = s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Multiset α
    ⊢ Eq (Union.union s 0) s
  -/
  simp [union_def]
  /-
    🎉 no goals
  -/


/-- `s ∩ t` is the lattice meet operation with respect to the
  multiset `≤`. The multiplicity of `a` in `s ∩ t` is the minimum
  of the multiplicities in `s` and `t`. -/
def inter (s t : Multiset α) : Multiset α :=
  Quotient.liftOn₂ s t (fun l₁ l₂ => (l₁.bagInter l₂ : Multiset α)) fun _v₁ _v₂ _w₁ _w₂ p₁ p₂ =>
    Quot.sound <| p₁.bagInter p₂


instance : Inter (Multiset α) :=
  ⟨inter⟩


@[simp]
theorem inter_zero (s : Multiset α) : s ∩ 0 = 0 :=
  Quot.inductionOn s fun l => congr_arg ofList l.bagInter_nil


@[simp]
theorem zero_inter (s : Multiset α) : 0 ∩ s = 0 :=
  Quot.inductionOn s fun l => congr_arg ofList l.nil_bagInter


@[simp]
theorem cons_inter_of_pos {a} (s : Multiset α) {t} : a ∈ t → (a ::ₘ s) ∩ t = a ::ₘ s ∩ t.erase a :=
  Quotient.inductionOn₂ s t fun _l₁ _l₂ h => congr_arg ofList <| cons_bagInter_of_pos _ h


@[simp]
theorem cons_inter_of_neg {a} (s : Multiset α) {t} : a ∉ t → (a ::ₘ s) ∩ t = s ∩ t :=
  Quotient.inductionOn₂ s t fun _l₁ _l₂ h => congr_arg ofList <| cons_bagInter_of_neg _ h


theorem inter_le_left {s t : Multiset α} : s ∩ t ≤ s :=
  Quotient.inductionOn₂ s t fun _l₁ _l₂ => (bagInter_sublist_left _ _).subperm


theorem inter_le_right {s t : Multiset α} : s ∩ t ≤ t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    ⊢ LE.le (Inter.inter s t) t
  -/
  induction' s using Multiset.induction_on with a s IH generalizing t
    /-
      case empty
      α : Type u_1
      inst✝ : DecidableEq α
      t : Multiset α
      ⊢ LE.le (Inter.inter 0 t) t
    -/
  · exact (zero_inter t).symm ▸ zero_le _
    /-
      🎉 no goals
    -/
  /-
    case cons
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s : Multiset α
    IH : ∀ {t : Multiset α}, LE.le (Inter.inter s t) t
    t : Multiset α
    ⊢ LE.le (Inter.inter (Multiset.cons a s) t) t
  -/
  by_cases h : a ∈ t
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      s : Multiset α
      IH : ∀ {t : Multiset α}, LE.le (Inter.inter s t) t
      t : Multiset α
      h : Membership.mem t a
      ⊢ LE.le (Inter.inter (Multiset.cons a s) t) t
    -/
  · simpa [h] using cons_le_cons a (IH (t := t.erase a))
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      s : Multiset α
      IH : ∀ {t : Multiset α}, LE.le (Inter.inter s t) t
      t : Multiset α
      h : Not (Membership.mem t a)
      ⊢ LE.le (Inter.inter (Multiset.cons a s) t) t
    -/
  · simp [h, IH]
    /-
      🎉 no goals
    -/


theorem le_inter (h₁ : s ≤ t) (h₂ : s ≤ u) : s ≤ t ∩ u := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t u : Multiset α
    h₁ : LE.le s t
    h₂ : LE.le s u
    ⊢ LE.le s (Inter.inter t u)
  -/
  revert s u; refine @(Multiset.induction_on t ?_ fun a t IH => ?_) <;> intros s u h₁ h₂
    /-
      case refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      t s u : Multiset α
      h₁ : LE.le s 0
      h₂ : LE.le s u
      ⊢ LE.le s (Inter.inter 0 u)
    -/
  · simpa only [zero_inter] using h₁
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    inst✝ : DecidableEq α
    t✝ : Multiset α
    a : α
    t : Multiset α
    IH : ∀ {s u : Multiset α}, LE.le s t → LE.le s u → LE.le s (Inter.inter t u)
    s u : Multiset α
    h₁ : LE.le s (Multiset.cons a t)
    h₂ : LE.le s u
    ⊢ LE.le s (Inter.inter (Multiset.cons a t) u)
  -/
  by_cases h : a ∈ u
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      t✝ : Multiset α
      a : α
      t : Multiset α
      IH : ∀ {s u : Multiset α}, LE.le s t → LE.le s u → LE.le s (Inter.inter t u)
      s u : Multiset α
      h₁ : LE.le s (Multiset.cons a t)
      h₂ : LE.le s u
      h : Membership.mem u a
      ⊢ LE.le s (Inter.inter (Multiset.cons a t) u)
    -/
  · rw [cons_inter_of_pos _ h, ← erase_le_iff_le_cons]
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      t✝ : Multiset α
      a : α
      t : Multiset α
      IH : ∀ {s u : Multiset α}, LE.le s t → LE.le s u → LE.le s (Inter.inter t u)
      s u : Multiset α
      h₁ : LE.le s (Multiset.cons a t)
      h₂ : LE.le s u
      h : Membership.mem u a
      ⊢ LE.le (s.erase a) (Inter.inter t (u.erase a))
    -/
    exact IH (erase_le_iff_le_cons.2 h₁) (erase_le_erase _ h₂)
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      t✝ : Multiset α
      a : α
      t : Multiset α
      IH : ∀ {s u : Multiset α}, LE.le s t → LE.le s u → LE.le s (Inter.inter t u)
      s u : Multiset α
      h₁ : LE.le s (Multiset.cons a t)
      h₂ : LE.le s u
      h : Not (Membership.mem u a)
      ⊢ LE.le s (Inter.inter (Multiset.cons a t) u)
    -/
  · rw [cons_inter_of_neg _ h]
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      t✝ : Multiset α
      a : α
      t : Multiset α
      IH : ∀ {s u : Multiset α}, LE.le s t → LE.le s u → LE.le s (Inter.inter t u)
      s u : Multiset α
      h₁ : LE.le s (Multiset.cons a t)
      h₂ : LE.le s u
      h : Not (Membership.mem u a)
      ⊢ LE.le s (Inter.inter t u)
    -/
    exact IH ((le_cons_of_not_mem <| mt (mem_of_le h₂) h).1 h₁) h₂
    /-
      🎉 no goals
    -/


@[simp]
theorem mem_inter : a ∈ s ∩ t ↔ a ∈ s ∧ a ∈ t :=
  ⟨fun h => ⟨mem_of_le inter_le_left h, mem_of_le inter_le_right h⟩, fun ⟨h₁, h₂⟩ => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Multiset α
      a : α
      x✝ : And (Membership.mem s a) (Membership.mem t a)
      h₁ : Membership.mem s a
      h₂ : Membership.mem t a
      ⊢ Membership.mem (Inter.inter s t) a
    -/
    rw [← cons_erase h₁, cons_inter_of_pos _ h₂]; apply mem_cons_self⟩
                                                  /-
                                                    🎉 no goals
                                                  -/


instance : Lattice (Multiset α) where
  sup := (· ∪ ·)
  sup_le _ _ _ := union_le
  le_sup_left _ _ := le_union_left
  le_sup_right _ _ := le_union_right
  inf := (· ∩ ·)
  le_inf _ _ _ := le_inter
  inf_le_left _ _ := inter_le_left
  inf_le_right _ _ := inter_le_right


@[simp]
theorem sup_eq_union (s t : Multiset α) : s ⊔ t = s ∪ t :=
  rfl


@[simp]
theorem inf_eq_inter (s t : Multiset α) : s ⊓ t = s ∩ t :=
  rfl


@[simp]
theorem le_inter_iff : s ≤ t ∩ u ↔ s ≤ t ∧ s ≤ u :=
  le_inf_iff


@[simp]
theorem union_le_iff : s ∪ t ≤ u ↔ s ≤ u ∧ t ≤ u :=
  sup_le_iff


theorem union_comm (s t : Multiset α) : s ∪ t = t ∪ s := sup_comm _ _


theorem inter_comm (s t : Multiset α) : s ∩ t = t ∩ s := inf_comm _ _


                                                     /-
                                                       α : Type u_1
                                                       inst✝ : DecidableEq α
                                                       s t : Multiset α
                                                       h : LE.le s t
                                                       ⊢ Eq (Union.union s t) t
                                                     -/
theorem eq_union_right (h : s ≤ t) : s ∪ t = t := by rw [union_comm, eq_union_left h]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[gcongr]
theorem union_le_union_left (h : s ≤ t) (u) : u ∪ s ≤ u ∪ t :=
  sup_le_sup_left h _


theorem union_le_add (s t : Multiset α) : s ∪ t ≤ s + t :=
  union_le (le_add_right _ _) (le_add_left _ _)


theorem union_add_distrib (s t u : Multiset α) : s ∪ t + u = s + u ∪ (t + u) := by
  simpa [(· ∪ ·), union, eq_comm, add_assoc] using
    show s + u - (t + u) = s - t by rw [add_comm t, tsub_add_eq_tsub_tsub, add_tsub_cancel_right]


theorem add_union_distrib (s t u : Multiset α) : s + (t ∪ u) = s + t ∪ (s + u) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t u : Multiset α
    ⊢ Eq (HAdd.hAdd s (Union.union t u)) (Union.union (HAdd.hAdd s t) (HAdd.hAdd s …
  -/
  rw [add_comm, union_add_distrib, add_comm s, add_comm s]
  /-
    🎉 no goals
  -/


theorem cons_union_distrib (a : α) (s t : Multiset α) : a ::ₘ (s ∪ t) = a ::ₘ s ∪ a ::ₘ t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s t : Multiset α
    ⊢ Eq (Multiset.cons a (Union.union s t)) (Union.union (Multiset.cons a s) (Mul …
  -/
  simpa using add_union_distrib (a ::ₘ 0) s t
  /-
    🎉 no goals
  -/


theorem inter_add_distrib (s t u : Multiset α) : s ∩ t + u = (s + u) ∩ (t + u) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t u : Multiset α
    ⊢ Eq (HAdd.hAdd (Inter.inter s t) u) (Inter.inter (HAdd.hAdd s u) (HAdd.hAdd t …
  -/
  by_contra! h
  obtain ⟨a, ha⟩ := lt_iff_cons_le.1 <| h.lt_of_le <| le_inter
    (add_le_add_right inter_le_left _) (add_le_add_right inter_le_right _)
  /-
    case intro
    α : Type u_1
    inst✝ : DecidableEq α
    s t u : Multiset α
    h : Ne (HAdd.hAdd (Inter.inter s t) u) (Inter.inter (HAdd.hAdd s u) (HAdd.hAdd …
    a : α
    ha : LE.le (Multiset.cons a (HAdd.hAdd (Inter.inter s t) u)) (Inter.inter (HAd …
    ⊢ False
  -/
  rw [← cons_add] at ha
  exact (lt_cons_self (s ∩ t) a).not_le <| le_inter
    (le_of_add_le_add_right (ha.trans inter_le_left))
    (le_of_add_le_add_right (ha.trans inter_le_right))


theorem add_inter_distrib (s t u : Multiset α) : s + t ∩ u = (s + t) ∩ (s + u) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t u : Multiset α
    ⊢ Eq (HAdd.hAdd s (Inter.inter t u)) (Inter.inter (HAdd.hAdd s t) (HAdd.hAdd s …
  -/
  rw [add_comm, inter_add_distrib, add_comm s, add_comm s]
  /-
    🎉 no goals
  -/


theorem cons_inter_distrib (a : α) (s t : Multiset α) : a ::ₘ s ∩ t = (a ::ₘ s) ∩ (a ::ₘ t) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s t : Multiset α
    ⊢ Eq (Multiset.cons a (Inter.inter s t)) (Inter.inter (Multiset.cons a s) (Mul …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma union_add_inter (s t : Multiset α) : s ∪ t + s ∩ t = s + t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    ⊢ Eq (HAdd.hAdd (Union.union s t) (Inter.inter s t)) (HAdd.hAdd s t)
  -/
  apply _root_.le_antisymm
    /-
      case a
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Multiset α
      ⊢ LE.le (HAdd.hAdd (Union.union s t) (Inter.inter s t)) (HAdd.hAdd s t)
    -/
  · rw [union_add_distrib]
    /-
      case a
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Multiset α
      ⊢ LE.le (Union.union (HAdd.hAdd s (Inter.inter s t)) (HAdd.hAdd t (Inter.inter …
    -/
    refine union_le (add_le_add_left inter_le_right _) ?_
    /-
      case a
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Multiset α
      ⊢ LE.le (HAdd.hAdd t (Inter.inter s t)) (HAdd.hAdd s t)
    -/
    rw [add_comm]
    /-
      case a
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Multiset α
      ⊢ LE.le (HAdd.hAdd (Inter.inter s t) t) (HAdd.hAdd s t)
    -/
    exact add_le_add_right inter_le_left _
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Multiset α
      ⊢ LE.le (HAdd.hAdd s t) (HAdd.hAdd (Union.union s t) (Inter.inter s t))
    -/
  · rw [add_comm, add_inter_distrib]
    /-
      case a
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Multiset α
      ⊢ LE.le (HAdd.hAdd t s) (Inter.inter (HAdd.hAdd (Union.union s t) s) (HAdd.hAd …
    -/
    refine le_inter (add_le_add_right le_union_right _) ?_
    /-
      case a
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Multiset α
      ⊢ LE.le (HAdd.hAdd t s) (HAdd.hAdd (Union.union s t) t)
    -/
    rw [add_comm]
    /-
      case a
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Multiset α
      ⊢ LE.le (HAdd.hAdd s t) (HAdd.hAdd (Union.union s t) t)
    -/
    exact add_le_add_right le_union_left _
    /-
      🎉 no goals
    -/


theorem sub_add_inter (s t : Multiset α) : s - t + s ∩ t = s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    ⊢ Eq (HAdd.hAdd (HSub.hSub s t) (Inter.inter s t)) s
  -/
  rw [inter_comm]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Multiset α
    ⊢ Eq (HAdd.hAdd (HSub.hSub s t) (Inter.inter t s)) s
  -/
  revert s; refine Multiset.induction_on t (by simp) fun a t IH s => ?_
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    t✝ : Multiset α
    a : α
    t : Multiset α
    IH : ∀ (s : Multiset α), Eq (HAdd.hAdd (HSub.hSub s t) (Inter.inter t s)) s
    s : Multiset α
    ⊢ Eq (HAdd.hAdd (HSub.hSub s (Multiset.cons a t)) (Inter.inter (Multiset.cons  …
  -/
  by_cases h : a ∈ s
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      t✝ : Multiset α
      a : α
      t : Multiset α
      IH : ∀ (s : Multiset α), Eq (HAdd.hAdd (HSub.hSub s t) (Inter.inter t s)) s
      s : Multiset α
      h : Membership.mem s a
      ⊢ Eq (HAdd.hAdd (HSub.hSub s (Multiset.cons a t)) (Inter.inter (Multiset.cons  …
    -/
  · rw [cons_inter_of_pos _ h, sub_cons, add_cons, IH, cons_erase h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      t✝ : Multiset α
      a : α
      t : Multiset α
      IH : ∀ (s : Multiset α), Eq (HAdd.hAdd (HSub.hSub s t) (Inter.inter t s)) s
      s : Multiset α
      h : Not (Membership.mem s a)
      ⊢ Eq (HAdd.hAdd (HSub.hSub s (Multiset.cons a t)) (Inter.inter (Multiset.cons  …
    -/
  · rw [cons_inter_of_neg _ h, sub_cons, erase_of_not_mem h, IH]
    /-
      🎉 no goals
    -/


theorem sub_inter (s t : Multiset α) : s - s ∩ t = s - t :=
  add_right_cancel (b := s ∩ t) <| by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Multiset α
      ⊢ Eq (HAdd.hAdd (HSub.hSub s (Inter.inter s t)) (Inter.inter s t)) (HAdd.hAdd  …
    -/
    rw [sub_add_inter s t, tsub_add_cancel_of_le inter_le_left]
    /-
      🎉 no goals
    -/


/-- `Filter p s` returns the elements in `s` (with the same multiplicities)
  which satisfy `p`, and removes the rest. -/
def filter (s : Multiset α) : Multiset α :=
  Quot.liftOn s (fun l => (List.filter p l : Multiset α)) fun _l₁ _l₂ h => Quot.sound <| h.filter p


@[simp, norm_cast] lemma filter_coe (l : List α) : filter p l = l.filter p := rfl


@[simp]
theorem filter_zero : filter p 0 = 0 :=
  rfl


@[congr]
theorem filter_congr {p q : α → Prop} [DecidablePred p] [DecidablePred q] {s : Multiset α} :
    (∀ x ∈ s, p x ↔ q x) → filter p s = filter q s :=
                                                                             /-
                                                                               α : Type u_1
                                                                               p q : α → Prop
                                                                               inst✝¹ : DecidablePred p
                                                                               inst✝ : DecidablePred q
                                                                               s : Multiset α
                                                                               _l : List α
                                                                               h : ∀ (x : α), Membership.mem (Quot.mk (⇑(List.isSetoid α)) _l) x → Iff (p x)  …
                                                                               ⊢ ∀ (x : α), Membership.mem _l x → Eq (Decidable.decide (p x)) (Decidable.deci …
                                                                             -/
  Quot.inductionOn s fun _l h => congr_arg ofList <| List.filter_congr <| by simpa using h
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


@[simp]
theorem filter_add (s t : Multiset α) : filter p (s + t) = filter p s + filter p t :=
  Quotient.inductionOn₂ s t fun _l₁ _l₂ => congr_arg ofList <| filter_append _ _


@[simp]
theorem filter_le (s : Multiset α) : filter p s ≤ s :=
  Quot.inductionOn s fun _l => (filter_sublist _).subperm


@[simp]
theorem filter_subset (s : Multiset α) : filter p s ⊆ s :=
  subset_of_le <| filter_le _ _


@[gcongr]
theorem filter_le_filter {s t} (h : s ≤ t) : filter p s ≤ filter p t :=
  leInductionOn h fun h => (h.filter (p ·)).subperm


theorem monotone_filter_left : Monotone (filter p) := fun _s _t => filter_le_filter p


theorem monotone_filter_right (s : Multiset α) ⦃p q : α → Prop⦄ [DecidablePred p] [DecidablePred q]
    (h : ∀ b, p b → q b) :
    s.filter p ≤ s.filter q :=
                                                                 /-
                                                                   α : Type u_1
                                                                   s : Multiset α
                                                                   p q : α → Prop
                                                                   inst✝¹ : DecidablePred p
                                                                   inst✝ : DecidablePred q
                                                                   h : ∀ (b : α), p b → q b
                                                                   l : List α
                                                                   ⊢ ∀ (a : α), Eq (Decidable.decide (p a)) Bool.true → Eq (Decidable.decide (q a …
                                                                 -/
  Quotient.inductionOn s fun l => (l.monotone_filter_right <| by simpa using h).subperm
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
theorem filter_cons_of_pos {a : α} (s) : p a → filter p (a ::ₘ s) = a ::ₘ filter p s :=
                                                                                  /-
                                                                                    α : Type u_1
                                                                                    p : α → Prop
                                                                                    inst✝ : DecidablePred p
                                                                                    a : α
                                                                                    s : Multiset α
                                                                                    x✝ : List α
                                                                                    h : p a
                                                                                    ⊢ Eq (Decidable.decide (p a)) Bool.true
                                                                                  -/
  Quot.inductionOn s fun _ h => congr_arg ofList <| List.filter_cons_of_pos <| by simpa using h
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[simp]
theorem filter_cons_of_neg {a : α} (s) : ¬p a → filter p (a ::ₘ s) = filter p s :=
                                                                                  /-
                                                                                    α : Type u_1
                                                                                    p : α → Prop
                                                                                    inst✝ : DecidablePred p
                                                                                    a : α
                                                                                    s : Multiset α
                                                                                    x✝ : List α
                                                                                    h : Not (p a)
                                                                                    ⊢ Not (Eq (Decidable.decide (p a)) Bool.true)
                                                                                  -/
  Quot.inductionOn s fun _ h => congr_arg ofList <| List.filter_cons_of_neg <| by simpa using h
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[simp]
theorem mem_filter {a : α} {s} : a ∈ filter p s ↔ a ∈ s ∧ p a :=
                                  /-
                                    α : Type u_1
                                    p : α → Prop
                                    inst✝ : DecidablePred p
                                    a : α
                                    s : Multiset α
                                    _l : List α
                                    ⊢ Iff (Membership.mem (Multiset.filter p (Quot.mk (⇑(List.isSetoid α)) _l)) a) …
                                  -/
  Quot.inductionOn s fun _l => by simp
                                  /-
                                    🎉 no goals
                                  -/


theorem of_mem_filter {a : α} {s} (h : a ∈ filter p s) : p a :=
  (mem_filter.1 h).2


theorem mem_of_mem_filter {a : α} {s} (h : a ∈ filter p s) : a ∈ s :=
  (mem_filter.1 h).1


theorem mem_filter_of_mem {a : α} {l} (m : a ∈ l) (h : p a) : a ∈ filter p l :=
  mem_filter.2 ⟨m, h⟩


theorem filter_eq_self {s} : filter p s = s ↔ ∀ a ∈ s, p a :=
  Quot.inductionOn s fun _l =>
    Iff.trans ⟨fun h => (filter_sublist _).eq_of_length (congr_arg card h),
                              /-
                                α : Type u_1
                                p : α → Prop
                                inst✝ : DecidablePred p
                                s : Multiset α
                                _l : List α
                                ⊢ Iff (Eq (List.filter (fun b => Decidable.decide (p b)) _l) _l) (∀ (a : α), M …
                              -/
      congr_arg ofList⟩ <| by simp
                              /-
                                🎉 no goals
                              -/


theorem filter_eq_nil {s} : filter p s = 0 ↔ ∀ a ∈ s, ¬p a :=
  Quot.inductionOn s fun _l =>
                                                                                           /-
                                                                                             α : Type u_1
                                                                                             p : α → Prop
                                                                                             inst✝ : DecidablePred p
                                                                                             s : Multiset α
                                                                                             _l : List α
                                                                                             ⊢ Iff (Eq (List.filter (fun b => Decidable.decide (p b)) _l) List.nil) (∀ (a : …
                                                                                           -/
    Iff.trans ⟨fun h => eq_nil_of_length_eq_zero (congr_arg card h), congr_arg ofList⟩ (by simp)
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


theorem le_filter {s t} : s ≤ filter p t ↔ s ≤ t ∧ ∀ a ∈ s, p a :=
  ⟨fun h => ⟨le_trans h (filter_le _ _), fun _a m => of_mem_filter (mem_of_le h m)⟩, fun ⟨h, al⟩ =>
    filter_eq_self.2 al ▸ filter_le_filter p h⟩


theorem filter_cons {a : α} (s : Multiset α) :
    filter p (a ::ₘ s) = (if p a then {a} else 0) + filter p s := by
  /-
    α : Type u_1
    p : α → Prop
    inst✝ : DecidablePred p
    a : α
    s : Multiset α
    ⊢ Eq (Multiset.filter p (Multiset.cons a s)) (HAdd.hAdd (ite (p a) (Singleton. …
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_1
      p : α → Prop
      inst✝ : DecidablePred p
      a : α
      s : Multiset α
      h : p a
      ⊢ Eq (Multiset.filter p (Multiset.cons a s)) (HAdd.hAdd (Singleton.singleton a …
    -/
  · rw [filter_cons_of_pos _ h, singleton_add]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      p : α → Prop
      inst✝ : DecidablePred p
      a : α
      s : Multiset α
      h : Not (p a)
      ⊢ Eq (Multiset.filter p (Multiset.cons a s)) (HAdd.hAdd 0 (Multiset.filter p s))
    -/
  · rw [filter_cons_of_neg _ h, zero_add]
    /-
      🎉 no goals
    -/


theorem filter_singleton {a : α} (p : α → Prop) [DecidablePred p] :
    filter p {a} = if p a then {a} else ∅ := by
  /-
    α : Type u_1
    a : α
    p : α → Prop
    inst✝ : DecidablePred p
    ⊢ Eq (Multiset.filter p (Singleton.singleton a)) (ite (p a) (Singleton.singlet …
  -/
  simp only [singleton, filter_cons, filter_zero, add_zero, empty_eq_zero]
  /-
    🎉 no goals
  -/


theorem filter_nsmul (s : Multiset α) (n : ℕ) : filter p (n • s) = n • filter p s := by
  /-
    α : Type u_1
    p : α → Prop
    inst✝ : DecidablePred p
    s : Multiset α
    n : Nat
    ⊢ Eq (Multiset.filter p (HSMul.hSMul n s)) (HSMul.hSMul n (Multiset.filter p s))
  -/
  refine s.induction_on ?_ ?_
    /-
      case refine_1
      α : Type u_1
      p : α → Prop
      inst✝ : DecidablePred p
      s : Multiset α
      n : Nat
      ⊢ Eq (Multiset.filter p (HSMul.hSMul n 0)) (HSMul.hSMul n (Multiset.filter p 0))
    -/
  · simp only [filter_zero, nsmul_zero]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      p : α → Prop
      inst✝ : DecidablePred p
      s : Multiset α
      n : Nat
      ⊢ ∀ (a : α) (s : Multiset α), Eq (Multiset.filter p (HSMul.hSMul n s)) (HSMul. …
    -/
  · intro a ha ih
    /-
      case refine_2
      α : Type u_1
      p : α → Prop
      inst✝ : DecidablePred p
      s : Multiset α
      n : Nat
      a : α
      ha : Multiset α
      ih : Eq (Multiset.filter p (HSMul.hSMul n ha)) (HSMul.hSMul n (Multiset.filter …
      ⊢ Eq (Multiset.filter p (HSMul.hSMul n (Multiset.cons a ha))) (HSMul.hSMul n ( …
    -/
    rw [nsmul_cons, filter_add, ih, filter_cons, nsmul_add]
    /-
      case refine_2
      α : Type u_1
      p : α → Prop
      inst✝ : DecidablePred p
      s : Multiset α
      n : Nat
      a : α
      ha : Multiset α
      ih : Eq (Multiset.filter p (HSMul.hSMul n ha)) (HSMul.hSMul n (Multiset.filter …
      ⊢ Eq (HAdd.hAdd (Multiset.filter p (HSMul.hSMul n (Singleton.singleton a))) (H …
    -/
    congr
    /-
      case refine_2.e_a
      α : Type u_1
      p : α → Prop
      inst✝ : DecidablePred p
      s : Multiset α
      n : Nat
      a : α
      ha : Multiset α
      ih : Eq (Multiset.filter p (HSMul.hSMul n ha)) (HSMul.hSMul n (Multiset.filter …
      ⊢ Eq (Multiset.filter p (HSMul.hSMul n (Singleton.singleton a))) (HSMul.hSMul  …
    -/
    split_ifs with hp <;>
        /-
          case pos
          α : Type u_1
          p : α → Prop
          inst✝ : DecidablePred p
          s : Multiset α
          n : Nat
          a : α
          ha : Multiset α
          ih : Eq (Multiset.filter p (HSMul.hSMul n ha)) (HSMul.hSMul n (Multiset.filter …
          hp : p a
          ⊢ Eq (Multiset.filter p (HSMul.hSMul n (Singleton.singleton a))) (HSMul.hSMul  …
        -/
        /-
          case pos
          α : Type u_1
          p : α → Prop
          inst✝ : DecidablePred p
          s : Multiset α
          n : Nat
          a : α
          ha : Multiset α
          ih : Eq (Multiset.filter p (HSMul.hSMul n ha)) (HSMul.hSMul n (Multiset.filter …
          hp : p a
          ⊢ ∀ (a_1 : α), Membership.mem (HSMul.hSMul n (Singleton.singleton a)) a_1 → p  …
        -/
        /-
          case pos
          α : Type u_1
          p : α → Prop
          inst✝ : DecidablePred p
          s : Multiset α
          n : Nat
          a : α
          ha : Multiset α
          ih : Eq (Multiset.filter p (HSMul.hSMul n ha)) (HSMul.hSMul n (Multiset.filter …
          hp : p a
          b : α
          hb : Membership.mem (HSMul.hSMul n (Singleton.singleton a)) b
          ⊢ p b
        -/
        /-
          🎉 no goals
        -/
        intro b hb
        /-
          case neg
          α : Type u_1
          p : α → Prop
          inst✝ : DecidablePred p
          s : Multiset α
          n : Nat
          a : α
          ha : Multiset α
          ih : Eq (Multiset.filter p (HSMul.hSMul n ha)) (HSMul.hSMul n (Multiset.filter …
          hp : Not (p a)
          b : α
          hb : Membership.mem (HSMul.hSMul n (Singleton.singleton a)) b
          ⊢ Not (p b)
        -/
        rwa [mem_singleton.mp (mem_of_mem_nsmul hb)]
        /-
          🎉 no goals
        -/


@[simp]
theorem filter_sub [DecidableEq α] (s t : Multiset α) :
    filter p (s - t) = filter p s - filter p t := by
  /-
    α : Type u_1
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidableEq α
    s t : Multiset α
    ⊢ Eq (Multiset.filter p (HSub.hSub s t)) (HSub.hSub (Multiset.filter p s) (Mul …
  -/
  revert s; refine Multiset.induction_on t (by simp) fun a t IH s => ?_
  /-
    α : Type u_1
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidableEq α
    t✝ : Multiset α
    a : α
    t : Multiset α
    IH : ∀ (s : Multiset α), Eq (Multiset.filter p (HSub.hSub s t)) (HSub.hSub (Mu …
    s : Multiset α
    ⊢ Eq (Multiset.filter p (HSub.hSub s (Multiset.cons a t))) (HSub.hSub (Multise …
  -/
  rw [sub_cons, IH]
  /-
    α : Type u_1
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidableEq α
    t✝ : Multiset α
    a : α
    t : Multiset α
    IH : ∀ (s : Multiset α), Eq (Multiset.filter p (HSub.hSub s t)) (HSub.hSub (Mu …
    s : Multiset α
    ⊢ Eq (HSub.hSub (Multiset.filter p (s.erase a)) (Multiset.filter p t)) (HSub.h …
  -/
  by_cases h : p a
    /-
      case pos
      α : Type u_1
      p : α → Prop
      inst✝¹ : DecidablePred p
      inst✝ : DecidableEq α
      t✝ : Multiset α
      a : α
      t : Multiset α
      IH : ∀ (s : Multiset α), Eq (Multiset.filter p (HSub.hSub s t)) (HSub.hSub (Mu …
      s : Multiset α
      h : p a
      ⊢ Eq (HSub.hSub (Multiset.filter p (s.erase a)) (Multiset.filter p t)) (HSub.h …
    -/
  · rw [filter_cons_of_pos _ h, sub_cons]
    /-
      case pos
      α : Type u_1
      p : α → Prop
      inst✝¹ : DecidablePred p
      inst✝ : DecidableEq α
      t✝ : Multiset α
      a : α
      t : Multiset α
      IH : ∀ (s : Multiset α), Eq (Multiset.filter p (HSub.hSub s t)) (HSub.hSub (Mu …
      s : Multiset α
      h : p a
      ⊢ Eq (HSub.hSub (Multiset.filter p (s.erase a)) (Multiset.filter p t)) (HSub.h …
    -/
    congr
    /-
      case pos.e_a
      α : Type u_1
      p : α → Prop
      inst✝¹ : DecidablePred p
      inst✝ : DecidableEq α
      t✝ : Multiset α
      a : α
      t : Multiset α
      IH : ∀ (s : Multiset α), Eq (Multiset.filter p (HSub.hSub s t)) (HSub.hSub (Mu …
      s : Multiset α
      h : p a
      ⊢ Eq (Multiset.filter p (s.erase a)) ((Multiset.filter p s).erase a)
    -/
    by_cases m : a ∈ s
    · rw [← cons_inj_right a, ← filter_cons_of_pos _ h, cons_erase (mem_filter_of_mem m h),
        cons_erase m]
      /-
        case neg
        α : Type u_1
        p : α → Prop
        inst✝¹ : DecidablePred p
        inst✝ : DecidableEq α
        t✝ : Multiset α
        a : α
        t : Multiset α
        IH : ∀ (s : Multiset α), Eq (Multiset.filter p (HSub.hSub s t)) (HSub.hSub (Mu …
        s : Multiset α
        h : p a
        m : Not (Membership.mem s a)
        ⊢ Eq (Multiset.filter p (s.erase a)) ((Multiset.filter p s).erase a)
      -/
    · rw [erase_of_not_mem m, erase_of_not_mem (mt mem_of_mem_filter m)]
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      p : α → Prop
      inst✝¹ : DecidablePred p
      inst✝ : DecidableEq α
      t✝ : Multiset α
      a : α
      t : Multiset α
      IH : ∀ (s : Multiset α), Eq (Multiset.filter p (HSub.hSub s t)) (HSub.hSub (Mu …
      s : Multiset α
      h : Not (p a)
      ⊢ Eq (HSub.hSub (Multiset.filter p (s.erase a)) (Multiset.filter p t)) (HSub.h …
    -/
  · rw [filter_cons_of_neg _ h]
    /-
      case neg
      α : Type u_1
      p : α → Prop
      inst✝¹ : DecidablePred p
      inst✝ : DecidableEq α
      t✝ : Multiset α
      a : α
      t : Multiset α
      IH : ∀ (s : Multiset α), Eq (Multiset.filter p (HSub.hSub s t)) (HSub.hSub (Mu …
      s : Multiset α
      h : Not (p a)
      ⊢ Eq (HSub.hSub (Multiset.filter p (s.erase a)) (Multiset.filter p t)) (HSub.h …
    -/
    by_cases m : a ∈ s
    · rw [(by rw [filter_cons_of_neg _ h] : filter p (erase s a) = filter p (a ::ₘ erase s a)),
        cons_erase m]
      /-
        case neg
        α : Type u_1
        p : α → Prop
        inst✝¹ : DecidablePred p
        inst✝ : DecidableEq α
        t✝ : Multiset α
        a : α
        t : Multiset α
        IH : ∀ (s : Multiset α), Eq (Multiset.filter p (HSub.hSub s t)) (HSub.hSub (Mu …
        s : Multiset α
        h : Not (p a)
        m : Not (Membership.mem s a)
        ⊢ Eq (HSub.hSub (Multiset.filter p (s.erase a)) (Multiset.filter p t)) (HSub.h …
      -/
    · rw [erase_of_not_mem m]
      /-
        🎉 no goals
      -/


@[simp]
theorem filter_union [DecidableEq α] (s t : Multiset α) :
                                                     /-
                                                       α : Type u_1
                                                       p : α → Prop
                                                       inst✝¹ : DecidablePred p
                                                       inst✝ : DecidableEq α
                                                       s t : Multiset α
                                                       ⊢ Eq (Multiset.filter p (Union.union s t)) (Union.union (Multiset.filter p s)  …
                                                     -/
    filter p (s ∪ t) = filter p s ∪ filter p t := by simp [(· ∪ ·), union]
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem filter_inter [DecidableEq α] (s t : Multiset α) :
    filter p (s ∩ t) = filter p s ∩ filter p t :=
  le_antisymm (le_inter (filter_le_filter _ inter_le_left) (filter_le_filter _ inter_le_right)) <|
    le_filter.2 ⟨inf_le_inf (filter_le _ _) (filter_le _ _), fun _a h =>
      of_mem_filter (mem_of_le inter_le_left h)⟩


@[simp]
theorem filter_filter (q) [DecidablePred q] (s : Multiset α) :
    filter p (filter q s) = filter (fun a => p a ∧ q a) s :=
                                 /-
                                   α : Type u_1
                                   p : α → Prop
                                   inst✝¹ : DecidablePred p
                                   q : α → Prop
                                   inst✝ : DecidablePred q
                                   s : Multiset α
                                   l : List α
                                   ⊢ Eq (Multiset.filter p (Multiset.filter q (Quot.mk (⇑(List.isSetoid α)) l)))  …
                                 -/
  Quot.inductionOn s fun l => by simp
                                 /-
                                   🎉 no goals
                                 -/


lemma filter_comm (q) [DecidablePred q] (s : Multiset α) :
                                                        /-
                                                          α : Type u_1
                                                          p : α → Prop
                                                          inst✝¹ : DecidablePred p
                                                          q : α → Prop
                                                          inst✝ : DecidablePred q
                                                          s : Multiset α
                                                          ⊢ Eq (Multiset.filter p (Multiset.filter q s)) (Multiset.filter q (Multiset.fi …
                                                        -/
    filter p (filter q s) = filter q (filter p s) := by simp [and_comm]
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem filter_add_filter (q) [DecidablePred q] (s : Multiset α) :
    filter p s + filter q s = filter (fun a => p a ∨ q a) s + filter (fun a => p a ∧ q a) s :=
                                               /-
                                                 α : Type u_1
                                                 p : α → Prop
                                                 inst✝¹ : DecidablePred p
                                                 q : α → Prop
                                                 inst✝ : DecidablePred q
                                                 s✝ : Multiset α
                                                 a : α
                                                 s : Multiset α
                                                 IH : Eq (HAdd.hAdd (Multiset.filter p s) (Multiset.filter q s)) (HAdd.hAdd (Mu …
                                                 ⊢ Eq (HAdd.hAdd (Multiset.filter p (Multiset.cons a s)) (Multiset.filter q (Mu …
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
  Multiset.induction_on s rfl fun a s IH => by by_cases p a <;> by_cases q a <;> simp [*]
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


theorem filter_add_not (s : Multiset α) : filter p s + filter (fun a => ¬p a) s = s := by
  /-
    α : Type u_1
    p : α → Prop
    inst✝ : DecidablePred p
    s : Multiset α
    ⊢ Eq (HAdd.hAdd (Multiset.filter p s) (Multiset.filter (fun a => Not (p a)) s) …
  -/
  rw [filter_add_filter, filter_eq_self.2, filter_eq_nil.2]
    /-
      α : Type u_1
      p : α → Prop
      inst✝ : DecidablePred p
      s : Multiset α
      ⊢ Eq (HAdd.hAdd s 0) s
    -/
  · simp only [add_zero]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      p : α → Prop
      inst✝ : DecidablePred p
      s : Multiset α
      ⊢ ∀ (a : α), Membership.mem s a → Not (And (p a) (Not (p a)))
    -/
  · simp [Decidable.em, -Bool.not_eq_true, -not_and, not_and_or, or_comm]
    /-
      🎉 no goals
    -/
  · simp only [Bool.not_eq_true, decide_eq_true_eq, Bool.eq_false_or_eq_true,
      decide_true, implies_true, Decidable.em]


theorem filter_map (f : β → α) (s : Multiset β) : filter p (map f s) = map f (filter (p ∘ f) s) :=
                                 /-
                                   α : Type u_1
                                   β : Type v
                                   p : α → Prop
                                   inst✝ : DecidablePred p
                                   f : β → α
                                   s : Multiset β
                                   l : List β
                                   ⊢ Eq (Multiset.filter p (Multiset.map f (Quot.mk (⇑(List.isSetoid β)) l))) (Mu …
                                 -/
  Quot.inductionOn s fun l => by simp [List.filter_map]; rfl
                                                         /-
                                                           🎉 no goals
                                                         -/


@[deprecated (since := "2024-06-16")] alias map_filter := filter_map

-- TODO: rename to `map_filter` when the deprecated alias above is removed.

lemma map_filter' {f : α → β} (hf : Injective f) (s : Multiset α)
    [DecidablePred fun b => ∃ a, p a ∧ f a = b] :
    (s.filter p).map f = (s.map f).filter fun b => ∃ a, p a ∧ f a = b := by
  /-
    α : Type u_1
    β : Type v
    p : α → Prop
    inst✝¹ : DecidablePred p
    f : α → β
    hf : Function.Injective f
    s : Multiset α
    inst✝ : DecidablePred fun b => Exists fun a => And (p a) (Eq (f a) b)
    ⊢ Eq (Multiset.map f (Multiset.filter p s)) (Multiset.filter (fun b => Exists  …
  -/
  simp [comp_def, filter_map, hf.eq_iff]
  /-
    🎉 no goals
  -/


lemma card_filter_le_iff (s : Multiset α) (P : α → Prop) [DecidablePred P] (n : ℕ) :
    card (s.filter P) ≤ n ↔ ∀ s' ≤ s, n < card s' → ∃ a ∈ s', ¬ P a := by
  /-
    α : Type u_1
    s : Multiset α
    P : α → Prop
    inst✝ : DecidablePred P
    n : Nat
    ⊢ Iff (LE.le (Multiset.filter P s).card n) (∀ (s' : Multiset α), LE.le s' s →  …
  -/
  fconstructor
    /-
      case mp
      α : Type u_1
      s : Multiset α
      P : α → Prop
      inst✝ : DecidablePred P
      n : Nat
      ⊢ LE.le (Multiset.filter P s).card n → ∀ (s' : Multiset α), LE.le s' s → LT.lt …
    -/
  · intro H s' hs' s'_card
    /-
      case mp
      α : Type u_1
      s : Multiset α
      P : α → Prop
      inst✝ : DecidablePred P
      n : Nat
      H : LE.le (Multiset.filter P s).card n
      s' : Multiset α
      hs' : LE.le s' s
      s'_card : LT.lt n s'.card
      ⊢ Exists fun a => And (Membership.mem s' a) (Not (P a))
    -/
    by_contra! rid
    /-
      case mp
      α : Type u_1
      s : Multiset α
      P : α → Prop
      inst✝ : DecidablePred P
      n : Nat
      H : LE.le (Multiset.filter P s).card n
      s' : Multiset α
      hs' : LE.le s' s
      s'_card : LT.lt n s'.card
      rid : ∀ (a : α), Membership.mem s' a → P a
      ⊢ False
    -/
    have card := card_le_card (monotone_filter_left P hs') |>.trans H
    /-
      case mp
      α : Type u_1
      s : Multiset α
      P : α → Prop
      inst✝ : DecidablePred P
      n : Nat
      H : LE.le (Multiset.filter P s).card n
      s' : Multiset α
      hs' : LE.le s' s
      s'_card : LT.lt n s'.card
      rid : ∀ (a : α), Membership.mem s' a → P a
      card : LE.le (Multiset.filter P s').card n
      ⊢ False
    -/
    exact s'_card.not_le (filter_eq_self.mpr rid ▸ card)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      s : Multiset α
      P : α → Prop
      inst✝ : DecidablePred P
      n : Nat
      ⊢ (∀ (s' : Multiset α), LE.le s' s → LT.lt n s'.card → Exists fun a => And (Me …
    -/
  · contrapose!
    /-
      case mpr
      α : Type u_1
      s : Multiset α
      P : α → Prop
      inst✝ : DecidablePred P
      n : Nat
      ⊢ LT.lt n (Multiset.filter P s).card → Exists fun s' => And (LE.le s' s) (And  …
    -/
    exact fun H ↦ ⟨s.filter P, filter_le _ _, H, fun a ha ↦ (mem_filter.mp ha).2⟩
    /-
      🎉 no goals
    -/


/-- `filterMap f s` is a combination filter/map operation on `s`.
  The function `f : α → Option β` is applied to each element of `s`;
  if `f a` is `some b` then `b` is added to the result, otherwise
  `a` is removed from the resulting multiset. -/
def filterMap (f : α → Option β) (s : Multiset α) : Multiset β :=
  Quot.liftOn s (fun l => (List.filterMap f l : Multiset β))
    fun _l₁ _l₂ h => Quot.sound <| h.filterMap f


@[simp, norm_cast]
lemma filterMap_coe (f : α → Option β) (l : List α) : filterMap f l = l.filterMap f := rfl


@[simp]
theorem filterMap_zero (f : α → Option β) : filterMap f 0 = 0 :=
  rfl


@[simp]
theorem filterMap_cons_none {f : α → Option β} (a : α) (s : Multiset α) (h : f a = none) :
    filterMap f (a ::ₘ s) = filterMap f s :=
  Quot.inductionOn s fun _ => congr_arg ofList <| List.filterMap_cons_none h


@[simp]
theorem filterMap_cons_some (f : α → Option β) (a : α) (s : Multiset α) {b : β}
    (h : f a = some b) : filterMap f (a ::ₘ s) = b ::ₘ filterMap f s :=
  Quot.inductionOn s fun _ => congr_arg ofList <| List.filterMap_cons_some h


theorem filterMap_eq_map (f : α → β) : filterMap (some ∘ f) = map f :=
  funext fun s =>
    Quot.inductionOn s fun l => congr_arg ofList <| congr_fun (List.filterMap_eq_map f) l


theorem filterMap_eq_filter : filterMap (Option.guard p) = filter p :=
  funext fun s =>
    Quot.inductionOn s fun l => congr_arg ofList <| by
      /-
        α : Type u_1
        p : α → Prop
        inst✝ : DecidablePred p
        s : Multiset α
        l : List α
        ⊢ Eq (List.filterMap (Option.guard p) l) (List.filter (fun b => Decidable.deci …
      -/
      rw [← List.filterMap_eq_filter]
      /-
        α : Type u_1
        p : α → Prop
        inst✝ : DecidablePred p
        s : Multiset α
        l : List α
        ⊢ Eq (List.filterMap (Option.guard p) l) (List.filterMap (Option.guard fun x = …
      -/
      congr; funext a; simp
                       /-
                         🎉 no goals
                       -/


theorem filterMap_filterMap (f : α → Option β) (g : β → Option γ) (s : Multiset α) :
    filterMap g (filterMap f s) = filterMap (fun x => (f x).bind g) s :=
  Quot.inductionOn s fun l => congr_arg ofList <| List.filterMap_filterMap f g l


theorem map_filterMap (f : α → Option β) (g : β → γ) (s : Multiset α) :
    map g (filterMap f s) = filterMap (fun x => (f x).map g) s :=
  Quot.inductionOn s fun l => congr_arg ofList <| List.map_filterMap f g l


theorem filterMap_map (f : α → β) (g : β → Option γ) (s : Multiset α) :
    filterMap g (map f s) = filterMap (g ∘ f) s :=
  Quot.inductionOn s fun l => congr_arg ofList <| List.filterMap_map f g l


theorem filter_filterMap (f : α → Option β) (p : β → Prop) [DecidablePred p] (s : Multiset α) :
    filter p (filterMap f s) = filterMap (fun x => (f x).filter p) s :=
  Quot.inductionOn s fun l => congr_arg ofList <| List.filter_filterMap f p l


theorem filterMap_filter (f : α → Option β) (s : Multiset α) :
    filterMap f (filter p s) = filterMap (fun x => if p x then f x else none) s :=
                                                     /-
                                                       α : Type u_1
                                                       β : Type v
                                                       p : α → Prop
                                                       inst✝ : DecidablePred p
                                                       f : α → Option β
                                                       s : Multiset α
                                                       l : List α
                                                       ⊢ Eq (List.filterMap f (List.filter (fun b => Decidable.decide (p b)) l)) (Lis …
                                                     -/
  Quot.inductionOn s fun l => congr_arg ofList <| by simpa using List.filterMap_filter p f l
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem filterMap_some (s : Multiset α) : filterMap some s = s :=
  Quot.inductionOn s fun l => congr_arg ofList <| List.filterMap_some l


@[simp]
theorem mem_filterMap (f : α → Option β) (s : Multiset α) {b : β} :
    b ∈ filterMap f s ↔ ∃ a, a ∈ s ∧ f a = some b :=
  Quot.inductionOn s fun _ => List.mem_filterMap


theorem map_filterMap_of_inv (f : α → Option β) (g : β → α) (H : ∀ x : α, (f x).map g = some x)
    (s : Multiset α) : map g (filterMap f s) = s :=
  Quot.inductionOn s fun l => congr_arg ofList <| List.map_filterMap_of_inv f g H l


@[gcongr]
theorem filterMap_le_filterMap (f : α → Option β) {s t : Multiset α} (h : s ≤ t) :
    filterMap f s ≤ filterMap f t :=
  leInductionOn h fun h => (h.filterMap _).subperm


/-- `countP p s` counts the number of elements of `s` (with multiplicity) that
  satisfy `p`. -/
def countP (s : Multiset α) : ℕ :=
  Quot.liftOn s (List.countP p) fun _l₁ _l₂ => Perm.countP_eq (p ·)


@[simp]
theorem coe_countP (l : List α) : countP p l = l.countP p :=
  rfl


@[simp]
theorem countP_zero : countP p 0 = 0 :=
  rfl


@[simp]
theorem countP_cons_of_pos {a : α} (s) : p a → countP p (a ::ₘ s) = countP p s + 1 :=
                           /-
                             α : Type u_1
                             p : α → Prop
                             inst✝ : DecidablePred p
                             a : α
                             s : Multiset α
                             ⊢ ∀ (a_1 : List α), p a → Eq (Multiset.countP p (Multiset.cons a (Quot.mk (⇑(L …
                           -/
  Quot.inductionOn s <| by simpa using List.countP_cons_of_pos (p ·)
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem countP_cons_of_neg {a : α} (s) : ¬p a → countP p (a ::ₘ s) = countP p s :=
                           /-
                             α : Type u_1
                             p : α → Prop
                             inst✝ : DecidablePred p
                             a : α
                             s : Multiset α
                             ⊢ ∀ (a_1 : List α), Not (p a) → Eq (Multiset.countP p (Multiset.cons a (Quot.m …
                           -/
  Quot.inductionOn s <| by simpa using List.countP_cons_of_neg (p ·)
                           /-
                             🎉 no goals
                           -/


theorem countP_cons (b : α) (s) : countP p (b ::ₘ s) = countP p s + if p b then 1 else 0 :=
                           /-
                             α : Type u_1
                             p : α → Prop
                             inst✝ : DecidablePred p
                             b : α
                             s : Multiset α
                             ⊢ ∀ (a : List α), Eq (Multiset.countP p (Multiset.cons b (Quot.mk (⇑(List.isSe …
                           -/
  Quot.inductionOn s <| by simp [List.countP_cons]
                           /-
                             🎉 no goals
                           -/


theorem countP_eq_card_filter (s) : countP p s = card (filter p s) :=
  Quot.inductionOn s fun l => l.countP_eq_length_filter (p ·)


theorem countP_le_card (s) : countP p s ≤ card s :=
  Quot.inductionOn s fun _l => countP_le_length (p ·)


@[simp]
theorem countP_add (s t) : countP p (s + t) = countP p s + countP p t := by
  /-
    α : Type u_1
    p : α → Prop
    inst✝ : DecidablePred p
    s t : Multiset α
    ⊢ Eq (Multiset.countP p (HAdd.hAdd s t)) (HAdd.hAdd (Multiset.countP p s) (Mul …
  -/
  simp [countP_eq_card_filter]
  /-
    🎉 no goals
  -/


@[simp]
theorem countP_nsmul (s) (n : ℕ) : countP p (n • s) = n * countP p s := by
  /-
    α : Type u_1
    p : α → Prop
    inst✝ : DecidablePred p
    s : Multiset α
    n : Nat
    ⊢ Eq (Multiset.countP p (HSMul.hSMul n s)) (HMul.hMul n (Multiset.countP p s))
  -/
                  /-
                    🎉 no goals
                  -/
  induction n <;> simp [*, succ_nsmul, succ_mul, zero_nsmul]
                  /-
                    🎉 no goals
                  -/


theorem card_eq_countP_add_countP (s) : card s = countP p s + countP (fun x => ¬p x) s :=
                                 /-
                                   α : Type u_1
                                   p : α → Prop
                                   inst✝ : DecidablePred p
                                   s : Multiset α
                                   l : List α
                                   ⊢ Eq (Multiset.card (Quot.mk (⇑(List.isSetoid α)) l)) (HAdd.hAdd (Multiset.cou …
                                 -/
  Quot.inductionOn s fun l => by simp [l.length_eq_countP_add_countP p]
                                 /-
                                   🎉 no goals
                                 -/


/-- `countP p`, the number of elements of a multiset satisfying `p`, promoted to an
`AddMonoidHom`. -/
def countPAddMonoidHom : Multiset α →+ ℕ where
  toFun := countP p
  map_zero' := countP_zero _
  map_add' := countP_add _


@[simp]
theorem coe_countPAddMonoidHom : (countPAddMonoidHom p : Multiset α → ℕ) = countP p :=
  rfl


@[simp]
theorem countP_sub [DecidableEq α] {s t : Multiset α} (h : t ≤ s) :
    countP p (s - t) = countP p s - countP p t := by
  /-
    α : Type u_1
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidableEq α
    s t : Multiset α
    h : LE.le t s
    ⊢ Eq (Multiset.countP p (HSub.hSub s t)) (HSub.hSub (Multiset.countP p s) (Mul …
  -/
  simp [countP_eq_card_filter, h, filter_le_filter]
  /-
    🎉 no goals
  -/


@[gcongr]
theorem countP_le_of_le {s t} (h : s ≤ t) : countP p s ≤ countP p t := by
  /-
    α : Type u_1
    p : α → Prop
    inst✝ : DecidablePred p
    s t : Multiset α
    h : LE.le s t
    ⊢ LE.le (Multiset.countP p s) (Multiset.countP p t)
  -/
  simpa [countP_eq_card_filter] using card_le_card (filter_le_filter p h)
  /-
    🎉 no goals
  -/


@[simp]
theorem countP_filter (q) [DecidablePred q] (s : Multiset α) :
                                                                /-
                                                                  α : Type u_1
                                                                  p : α → Prop
                                                                  inst✝¹ : DecidablePred p
                                                                  q : α → Prop
                                                                  inst✝ : DecidablePred q
                                                                  s : Multiset α
                                                                  ⊢ Eq (Multiset.countP p (Multiset.filter q s)) (Multiset.countP (fun a => And  …
                                                                -/
    countP p (filter q s) = countP (fun a => p a ∧ q a) s := by simp [countP_eq_card_filter]
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem countP_eq_countP_filter_add (s) (p q : α → Prop) [DecidablePred p] [DecidablePred q] :
    countP p s = (filter q s).countP p + (filter (fun a => ¬q a) s).countP p :=
  Quot.inductionOn s fun l => by
    /-
      α : Type u_1
      s : Multiset α
      p q : α → Prop
      inst✝¹ : DecidablePred p
      inst✝ : DecidablePred q
      l : List α
      ⊢ Eq (Multiset.countP p (Quot.mk (⇑(List.isSetoid α)) l)) (HAdd.hAdd (Multiset …
    -/
    convert l.countP_eq_countP_filter_add (p ·) (q ·)
    /-
      case h.e'_3.h.e'_6
      α : Type u_1
      s : Multiset α
      p q : α → Prop
      inst✝¹ : DecidablePred p
      inst✝ : DecidablePred q
      l : List α
      ⊢ Eq (Multiset.countP p (Multiset.filter (fun a => Not (q a)) (Quot.mk (⇑(List …
    -/
    simp [countP_filter]
    /-
      🎉 no goals
    -/


@[simp]
theorem countP_True {s : Multiset α} : countP (fun _ => True) s = card s :=
  Quot.inductionOn s fun _l => congrFun List.countP_true _


@[simp]
theorem countP_False {s : Multiset α} : countP (fun _ => False) s = 0 :=
  Quot.inductionOn s fun _l => congrFun List.countP_false _


theorem countP_map (f : α → β) (s : Multiset α) (p : β → Prop) [DecidablePred p] :
    countP p (map f s) = card (s.filter fun a => p (f a)) := by
  /-
    α : Type u_1
    β : Type v
    f : α → β
    s : Multiset α
    p : β → Prop
    inst✝ : DecidablePred p
    ⊢ Eq (Multiset.countP p (Multiset.map f s)) (Multiset.filter (fun a => p (f a) …
  -/
  refine Multiset.induction_on s ?_ fun a t IH => ?_
    /-
      case refine_1
      α : Type u_1
      β : Type v
      f : α → β
      s : Multiset α
      p : β → Prop
      inst✝ : DecidablePred p
      ⊢ Eq (Multiset.countP p (Multiset.map f 0)) (Multiset.filter (fun a => p (f a) …
    -/
  · rw [map_zero, countP_zero, filter_zero, card_zero]
    /-
      🎉 no goals
    -/
  · rw [map_cons, countP_cons, IH, filter_cons, card_add, apply_ite card, card_zero, card_singleton,
      add_comm]

-- Porting note: `Lean.Internal.coeM` forces us to type-ascript `{a // a ∈ s}`

lemma countP_attach (s : Multiset α) : s.attach.countP (fun a : {a // a ∈ s} ↦ p a) = s.countP p :=
  Quotient.inductionOn s fun l => by
    /-
      α : Type u_1
      p : α → Prop
      inst✝ : DecidablePred p
      s : Multiset α
      l : List α
      ⊢ Eq (Multiset.countP (fun a => p ↑a) (Multiset.attach (Quotient.mk (List.isSe …
    -/
    simp only [quot_mk_to_coe, coe_countP]
    -- Porting note: was
    -- rw [quot_mk_to_coe, coe_attach, coe_countP]
    -- exact List.countP_attach _ _
    /-
      α : Type u_1
      p : α → Prop
      inst✝ : DecidablePred p
      s : Multiset α
      l : List α
      ⊢ Eq (Multiset.countP (fun a => p ↑a) (↑l).attach) (List.countP (fun b => Deci …
    -/
    rw [coe_attach]
    /-
      α : Type u_1
      p : α → Prop
      inst✝ : DecidablePred p
      s : Multiset α
      l : List α
      ⊢ Eq (Multiset.countP (fun a => p ↑a) ↑l.attach) (List.countP (fun b => Decida …
    -/
    refine (coe_countP _ _).trans ?_
    /-
      α : Type u_1
      p : α → Prop
      inst✝ : DecidablePred p
      s : Multiset α
      l : List α
      ⊢ Eq (List.countP (fun b => Decidable.decide (p ↑b)) l.attach) (List.countP (f …
    -/
    convert List.countP_attach _ _
    /-
      case h.e'_2
      α : Type u_1
      p : α → Prop
      inst✝ : DecidablePred p
      s : Multiset α
      l : List α
      ⊢ Eq (List.countP (fun b => Decidable.decide (p ↑b)) l.attach) (List.countP (f …
    -/
    rfl
    /-
      🎉 no goals
    -/


lemma filter_attach (s : Multiset α) (p : α → Prop) [DecidablePred p] :
    (s.attach.filter fun a : {a // a ∈ s} ↦ p ↑a) =
      (s.filter p).attach.map (Subtype.map id fun _ ↦ Multiset.mem_of_mem_filter) :=
  Quotient.inductionOn s fun l ↦ congr_arg _ (List.filter_attach l p)


theorem countP_pos {s} : 0 < countP p s ↔ ∃ a ∈ s, p a :=
                                  /-
                                    α : Type u_1
                                    p : α → Prop
                                    inst✝ : DecidablePred p
                                    s : Multiset α
                                    _l : List α
                                    ⊢ Iff (LT.lt 0 (Multiset.countP p (Quot.mk (⇑(List.isSetoid α)) _l))) (Exists  …
                                  -/
  Quot.inductionOn s fun _l => by simp
                                  /-
                                    🎉 no goals
                                  -/


theorem countP_eq_zero {s} : countP p s = 0 ↔ ∀ a ∈ s, ¬p a :=
                                  /-
                                    α : Type u_1
                                    p : α → Prop
                                    inst✝ : DecidablePred p
                                    s : Multiset α
                                    _l : List α
                                    ⊢ Iff (Eq (Multiset.countP p (Quot.mk (⇑(List.isSetoid α)) _l)) 0) (∀ (a : α), …
                                  -/
  Quot.inductionOn s fun _l => by simp [List.countP_eq_zero]
                                  /-
                                    🎉 no goals
                                  -/


theorem countP_eq_card {s} : countP p s = card s ↔ ∀ a ∈ s, p a :=
                                  /-
                                    α : Type u_1
                                    p : α → Prop
                                    inst✝ : DecidablePred p
                                    s : Multiset α
                                    _l : List α
                                    ⊢ Iff (Eq (Multiset.countP p (Quot.mk (⇑(List.isSetoid α)) _l)) (Multiset.card …
                                  -/
  Quot.inductionOn s fun _l => by simp [List.countP_eq_length]
                                  /-
                                    🎉 no goals
                                  -/


theorem countP_pos_of_mem {s a} (h : a ∈ s) (pa : p a) : 0 < countP p s :=
  countP_pos.2 ⟨_, h, pa⟩


@[congr]
theorem countP_congr {s s' : Multiset α} (hs : s = s')
    {p p' : α → Prop} [DecidablePred p] [DecidablePred p']
    (hp : ∀ x ∈ s, p x = p' x) : s.countP p = s'.countP p' := by
  /-
    α : Type u_1
    s s' : Multiset α
    hs : Eq s s'
    p p' : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidablePred p'
    hp : ∀ (x : α), Membership.mem s x → Eq (p x) (p' x)
    ⊢ Eq (Multiset.countP p s) (Multiset.countP p' s')
  -/
  revert hs hp
  exact Quot.induction_on₂ s s'
    (fun l l' hs hp => by
      simp only [quot_mk_to_coe'', coe_eq_coe] at hs
      apply hs.countP_congr
      simpa using hp)


/-- `count a s` is the multiplicity of `a` in `s`. -/
def count (a : α) : Multiset α → ℕ :=
  countP (a = ·)


@[simp]
theorem coe_count (a : α) (l : List α) : count a (ofList l) = l.count a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    l : List α
    ⊢ Eq (Multiset.count a ↑l) (List.count a l)
  -/
  simp_rw [count, List.count, coe_countP (a = ·) l, @eq_comm _ a]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    l : List α
    ⊢ Eq (List.countP (fun b => Decidable.decide (Eq b a)) l) (List.countP (fun x  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem count_zero (a : α) : count a 0 = 0 :=
  rfl


@[simp]
theorem count_cons_self (a : α) (s : Multiset α) : count a (a ::ₘ s) = count a s + 1 :=
  countP_cons_of_pos _ <| rfl


@[simp]
theorem count_cons_of_ne {a b : α} (h : a ≠ b) (s : Multiset α) : count a (b ::ₘ s) = count a s :=
  countP_cons_of_neg _ <| h


theorem count_le_card (a : α) (s) : count a s ≤ card s :=
  countP_le_card _ _


@[gcongr]
theorem count_le_of_le (a : α) {s t} : s ≤ t → count a s ≤ count a t :=
  countP_le_of_le _


theorem count_le_count_cons (a b : α) (s : Multiset α) : count a s ≤ count a (b ::ₘ s) :=
  count_le_of_le _ (le_cons_self _ _)


theorem count_cons (a b : α) (s : Multiset α) :
    count a (b ::ₘ s) = count a s + if a = b then 1 else 0 :=
  countP_cons (a = ·) _ _


theorem count_singleton_self (a : α) : count a ({a} : Multiset α) = 1 :=
  count_eq_one_of_mem (nodup_singleton a) <| mem_singleton_self a


theorem count_singleton (a b : α) : count a ({b} : Multiset α) = if a = b then 1 else 0 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a b : α
    ⊢ Eq (Multiset.count a (Singleton.singleton b)) (ite (Eq a b) 1 0)
  -/
  simp only [count_cons, ← cons_zero, count_zero, zero_add]
  /-
    🎉 no goals
  -/


@[simp]
theorem count_add (a : α) : ∀ s t, count a (s + t) = count a s + count a t :=
  countP_add _


/-- `count a`, the multiplicity of `a` in a multiset, promoted to an `AddMonoidHom`. -/
def countAddMonoidHom (a : α) : Multiset α →+ ℕ :=
  countPAddMonoidHom (a = ·)


@[simp]
theorem coe_countAddMonoidHom {a : α} : (countAddMonoidHom a : Multiset α → ℕ) = count a :=
  rfl


@[simp]
theorem count_nsmul (a : α) (n s) : count a (n • s) = n * count a s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    n : Nat
    s : Multiset α
    ⊢ Eq (Multiset.count a (HSMul.hSMul n s)) (HMul.hMul n (Multiset.count a s))
  -/
                  /-
                    🎉 no goals
                  -/
  induction n <;> simp [*, succ_nsmul, succ_mul, zero_nsmul]
                  /-
                    🎉 no goals
                  -/


@[simp]
lemma count_attach (a : {x // x ∈ s}) : s.attach.count a = s.count ↑a :=
                                           /-
                                             α : Type u_1
                                             inst✝ : DecidableEq α
                                             s : Multiset α
                                             a x✝¹ : Subtype fun x => Membership.mem s x
                                             x✝ : Membership.mem s.attach x✝¹
                                             ⊢ Eq (Eq a x✝¹) (Eq ↑a ↑x✝¹)
                                           -/
  Eq.trans (countP_congr rfl fun _ _ => by simp [Subtype.ext_iff]) <| countP_attach _ _
                                           /-
                                             🎉 no goals
                                           -/


                                                                         /-
                                                                           α : Type u_1
                                                                           inst✝ : DecidableEq α
                                                                           a : α
                                                                           s : Multiset α
                                                                           ⊢ Iff (LT.lt 0 (Multiset.count a s)) (Membership.mem s a)
                                                                         -/
theorem count_pos {a : α} {s : Multiset α} : 0 < count a s ↔ a ∈ s := by simp [count, countP_pos]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


theorem one_le_count_iff_mem {a : α} {s : Multiset α} : 1 ≤ count a s ↔ a ∈ s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s : Multiset α
    ⊢ Iff (LE.le 1 (Multiset.count a s)) (Membership.mem s a)
  -/
  rw [succ_le_iff, count_pos]
  /-
    🎉 no goals
  -/


@[simp]
theorem count_eq_zero_of_not_mem {a : α} {s : Multiset α} (h : a ∉ s) : count a s = 0 :=
  by_contradiction fun h' => h <| count_pos.1 (Nat.pos_of_ne_zero h')


lemma count_ne_zero {a : α} : count a s ≠ 0 ↔ a ∈ s := Nat.pos_iff_ne_zero.symm.trans count_pos


@[simp] lemma count_eq_zero {a : α} : count a s = 0 ↔ a ∉ s := count_ne_zero.not_right


theorem count_eq_card {a : α} {s} : count a s = card s ↔ ∀ x ∈ s, a = x := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s : Multiset α
    ⊢ Iff (Eq (Multiset.count a s) s.card) (∀ (x : α), Membership.mem s x → Eq a x)
  -/
  simp [countP_eq_card, count, @eq_comm _ a]
  /-
    🎉 no goals
  -/


@[simp]
theorem count_replicate_self (a : α) (n : ℕ) : count a (replicate n a) = n := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    n : Nat
    ⊢ Eq (Multiset.count a (Multiset.replicate n a)) n
  -/
  convert List.count_replicate_self a n
  /-
    case h.e'_2
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    n : Nat
    ⊢ Eq (Multiset.count a (Multiset.replicate n a)) (List.count a (List.replicate …
  -/
  rw [← coe_count, coe_replicate]
  /-
    🎉 no goals
  -/


theorem count_replicate (a b : α) (n : ℕ) : count a (replicate n b) = if b = a then n else 0 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a b : α
    n : Nat
    ⊢ Eq (Multiset.count a (Multiset.replicate n b)) (ite (Eq b a) n 0)
  -/
  convert List.count_replicate a b n
    /-
      case h.e'_2
      α : Type u_1
      inst✝ : DecidableEq α
      a b : α
      n : Nat
      ⊢ Eq (Multiset.count a (Multiset.replicate n b)) (List.count a (List.replicate …
    -/
  · rw [← coe_count, coe_replicate]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h₁.a
      α : Type u_1
      inst✝ : DecidableEq α
      a b : α
      n : Nat
      ⊢ Iff (Eq b a) (Eq (BEq.beq b a) Bool.true)
    -/
  · simp
    /-
      🎉 no goals
    -/


@[simp]
theorem count_erase_self (a : α) (s : Multiset α) : count a (erase s a) = count a s - 1 :=
  Quotient.inductionOn s fun l => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      s : Multiset α
      l : List α
      ⊢ Eq (Multiset.count a (Multiset.erase (Quotient.mk (List.isSetoid α) l) a)) ( …
    -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
    convert List.count_erase_self a l <;> rw [← coe_count] <;> simp
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp]
theorem count_erase_of_ne {a b : α} (ab : a ≠ b) (s : Multiset α) :
    count a (erase s b) = count a s :=
  Quotient.inductionOn s fun l => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      a b : α
      ab : Ne a b
      s : Multiset α
      l : List α
      ⊢ Eq (Multiset.count a (Multiset.erase (Quotient.mk (List.isSetoid α) l) b)) ( …
    -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
    convert List.count_erase_of_ne ab l <;> rw [← coe_count] <;> simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
theorem count_sub (a : α) (s t : Multiset α) : count a (s - t) = count a s - count a t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s t : Multiset α
    ⊢ Eq (Multiset.count a (HSub.hSub s t)) (HSub.hSub (Multiset.count a s) (Multi …
  -/
  revert s; refine Multiset.induction_on t (by simp) fun b t IH s => ?_
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    t✝ : Multiset α
    b : α
    t : Multiset α
    IH : ∀ (s : Multiset α), Eq (Multiset.count a (HSub.hSub s t)) (HSub.hSub (Mul …
    s : Multiset α
    ⊢ Eq (Multiset.count a (HSub.hSub s (Multiset.cons b t))) (HSub.hSub (Multiset …
  -/
  rw [sub_cons, IH]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    t✝ : Multiset α
    b : α
    t : Multiset α
    IH : ∀ (s : Multiset α), Eq (Multiset.count a (HSub.hSub s t)) (HSub.hSub (Mul …
    s : Multiset α
    ⊢ Eq (HSub.hSub (Multiset.count a (s.erase b)) (Multiset.count a t)) (HSub.hSu …
  -/
  rcases Decidable.eq_or_ne a b with rfl | ab
    /-
      case inl
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      t✝ t : Multiset α
      IH : ∀ (s : Multiset α), Eq (Multiset.count a (HSub.hSub s t)) (HSub.hSub (Mul …
      s : Multiset α
      ⊢ Eq (HSub.hSub (Multiset.count a (s.erase a)) (Multiset.count a t)) (HSub.hSu …
    -/
  · rw [count_erase_self, count_cons_self, Nat.sub_sub, add_comm]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      t✝ : Multiset α
      b : α
      t : Multiset α
      IH : ∀ (s : Multiset α), Eq (Multiset.count a (HSub.hSub s t)) (HSub.hSub (Mul …
      s : Multiset α
      ab : Ne a b
      ⊢ Eq (HSub.hSub (Multiset.count a (s.erase b)) (Multiset.count a t)) (HSub.hSu …
    -/
  · rw [count_erase_of_ne ab, count_cons_of_ne ab]
    /-
      🎉 no goals
    -/


@[simp]
theorem count_union (a : α) (s t : Multiset α) : count a (s ∪ t) = max (count a s) (count a t) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s t : Multiset α
    ⊢ Eq (Multiset.count a (Union.union s t)) (Max.max (Multiset.count a s) (Multi …
  -/
  simp [(· ∪ ·), union, Nat.sub_add_eq_max]
  /-
    🎉 no goals
  -/


@[simp]
theorem count_inter (a : α) (s t : Multiset α) : count a (s ∩ t) = min (count a s) (count a t) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s t : Multiset α
    ⊢ Eq (Multiset.count a (Inter.inter s t)) (Min.min (Multiset.count a s) (Multi …
  -/
  apply @Nat.add_left_cancel (count a (s - t))
  /-
    case a
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s t : Multiset α
    ⊢ Eq (HAdd.hAdd (Multiset.count a (HSub.hSub s t)) (Multiset.count a (Inter.in …
  -/
  rw [← count_add, sub_add_inter, count_sub, Nat.sub_add_min_cancel]
  /-
    🎉 no goals
  -/


theorem le_count_iff_replicate_le {a : α} {s : Multiset α} {n : ℕ} :
    n ≤ count a s ↔ replicate n a ≤ s :=
  Quot.inductionOn s fun _l => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      s : Multiset α
      n : Nat
      _l : List α
      ⊢ Iff (LE.le n (Multiset.count a (Quot.mk (⇑(List.isSetoid α)) _l))) (LE.le (M …
    -/
    simp only [quot_mk_to_coe'', mem_coe, coe_count]
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      s : Multiset α
      n : Nat
      _l : List α
      ⊢ Iff (LE.le n (List.count a _l)) (LE.le (Multiset.replicate n a) ↑_l)
    -/
    exact le_count_iff_replicate_sublist.trans replicate_le_coe.symm
    /-
      🎉 no goals
    -/


@[simp]
theorem count_filter_of_pos {p} [DecidablePred p] {a} {s : Multiset α} (h : p a) :
    count a (filter p s) = count a s :=
  Quot.inductionOn s fun _l => by
    /-
      α : Type u_1
      inst✝¹ : DecidableEq α
      p : α → Prop
      inst✝ : DecidablePred p
      a : α
      s : Multiset α
      h : p a
      _l : List α
      ⊢ Eq (Multiset.count a (Multiset.filter p (Quot.mk (⇑(List.isSetoid α)) _l)))  …
    -/
    simp only [quot_mk_to_coe'', filter_coe, mem_coe, coe_count, decide_eq_true_eq]
    /-
      α : Type u_1
      inst✝¹ : DecidableEq α
      p : α → Prop
      inst✝ : DecidablePred p
      a : α
      s : Multiset α
      h : p a
      _l : List α
      ⊢ Eq (List.count a (List.filter (fun b => Decidable.decide (p b)) _l)) (List.c …
    -/
    apply count_filter
    /-
      case h
      α : Type u_1
      inst✝¹ : DecidableEq α
      p : α → Prop
      inst✝ : DecidablePred p
      a : α
      s : Multiset α
      h : p a
      _l : List α
      ⊢ Eq (Decidable.decide (p a)) Bool.true
    -/
    simpa using h
    /-
      🎉 no goals
    -/


@[simp]
theorem count_filter_of_neg {p} [DecidablePred p] {a} {s : Multiset α} (h : ¬p a) :
    count a (filter p s) = 0 :=
  Multiset.count_eq_zero_of_not_mem fun t => h (of_mem_filter t)


theorem count_filter {p} [DecidablePred p] {a} {s : Multiset α} :
    count a (filter p s) = if p a then count a s else 0 := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    p : α → Prop
    inst✝ : DecidablePred p
    a : α
    s : Multiset α
    ⊢ Eq (Multiset.count a (Multiset.filter p s)) (ite (p a) (Multiset.count a s) 0)
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_1
      inst✝¹ : DecidableEq α
      p : α → Prop
      inst✝ : DecidablePred p
      a : α
      s : Multiset α
      h : p a
      ⊢ Eq (Multiset.count a (Multiset.filter p s)) (Multiset.count a s)
    -/
  · exact count_filter_of_pos h
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝¹ : DecidableEq α
      p : α → Prop
      inst✝ : DecidablePred p
      a : α
      s : Multiset α
      h : Not (p a)
      ⊢ Eq (Multiset.count a (Multiset.filter p s)) 0
    -/
  · exact count_filter_of_neg h
    /-
      🎉 no goals
    -/


theorem ext {s t : Multiset α} : s = t ↔ ∀ a, count a s = count a t :=
  Quotient.inductionOn₂ s t fun _l₁ _l₂ => Quotient.eq.trans <| by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Multiset α
      _l₁ _l₂ : List α
      ⊢ Iff ((List.isSetoid α) _l₁ _l₂) (∀ (a : α), Eq (Multiset.count a (Quotient.m …
    -/
    simp only [quot_mk_to_coe, filter_coe, mem_coe, coe_count, decide_eq_true_eq]
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Multiset α
      _l₁ _l₂ : List α
      ⊢ Iff ((List.isSetoid α) _l₁ _l₂) (∀ (a : α), Eq (List.count a _l₁) (List.coun …
    -/
    apply perm_iff_count
    /-
      🎉 no goals
    -/


@[ext]
theorem ext' {s t : Multiset α} : (∀ a, count a s = count a t) → s = t :=
  ext.2


lemma count_injective : Injective fun (s : Multiset α) a ↦ s.count a :=
  fun _s _t hst ↦ ext' <| congr_fun hst


@[simp]
                                                                                        /-
                                                                                          α : Type u_1
                                                                                          inst✝ : DecidableEq α
                                                                                          s t : List α
                                                                                          ⊢ Eq (Inter.inter ↑s ↑t) ↑(s.bagInter t)
                                                                                        -/
theorem coe_inter (s t : List α) : (s ∩ t : Multiset α) = (s.bagInter t : List α) := by ext; simp
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/


theorem le_iff_count {s t : Multiset α} : s ≤ t ↔ ∀ a, count a s ≤ count a t :=
  ⟨fun h a => count_le_of_le a h, fun al => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Multiset α
      al : ∀ (a : α), LE.le (Multiset.count a s) (Multiset.count a t)
      ⊢ LE.le s t
    -/
    rw [← (ext.2 fun a => by simp [max_eq_right (al a)] : s ∪ t = t)]; apply le_union_left⟩
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


instance : DistribLattice (Multiset α) :=
  { le_sup_inf := fun s t u =>
      le_of_eq <|
        Eq.symm <|
          ext.2 fun a => by
            simp only [max_min_distrib_left, Multiset.count_inter, Multiset.sup_eq_union,
              Multiset.count_union, Multiset.inf_eq_inter] }


theorem count_map {α β : Type*} (f : α → β) (s : Multiset α) [DecidableEq β] (b : β) :
    count b (map f s) = card (s.filter fun a => b = f a) := by
  /-
    α : Type u_3
    β : Type u_4
    f : α → β
    s : Multiset α
    inst✝ : DecidableEq β
    b : β
    ⊢ Eq (Multiset.count b (Multiset.map f s)) (Multiset.filter (fun a => Eq b (f  …
  -/
  simp [Bool.beq_eq_decide_eq, eq_comm, count, countP_map]
  /-
    🎉 no goals
  -/


/-- `Multiset.map f` preserves `count` if `f` is injective on the set of elements contained in
the multiset -/
theorem count_map_eq_count [DecidableEq β] (f : α → β) (s : Multiset α)
    (hf : Set.InjOn f { x : α | x ∈ s }) (x) (H : x ∈ s) : (s.map f).count (f x) = s.count x := by
  suffices (filter (fun a : α => f x = f a) s).count x = card (filter (fun a : α => f x = f a) s) by
    rw [count, countP_map, ← this]
    exact count_filter_of_pos <| rfl
    /-
      α : Type u_1
      β : Type v
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      f : α → β
      s : Multiset α
      hf : Set.InjOn f (setOf fun x => Membership.mem s x)
      x : α
      H : Membership.mem s x
      ⊢ Eq (Multiset.count x (Multiset.filter (fun a => Eq (f x) (f a)) s)) (Multise …
    -/
  · rw [eq_replicate_card.2 fun b hb => (hf H (mem_filter.1 hb).left _).symm]
      /-
        α : Type u_1
        β : Type v
        inst✝¹ : DecidableEq α
        inst✝ : DecidableEq β
        f : α → β
        s : Multiset α
        hf : Set.InjOn f (setOf fun x => Membership.mem s x)
        x : α
        H : Membership.mem s x
        ⊢ Eq (Multiset.count x (Multiset.replicate (Multiset.filter (fun a => Eq (f x) …
      -/
    · simp only [count_replicate, eq_self_iff_true, if_true, card_replicate]
      /-
        🎉 no goals
      -/
      /-
        α : Type u_1
        β : Type v
        inst✝¹ : DecidableEq α
        inst✝ : DecidableEq β
        f : α → β
        s : Multiset α
        hf : Set.InjOn f (setOf fun x => Membership.mem s x)
        x : α
        H : Membership.mem s x
        ⊢ ∀ (b : α), Membership.mem (Multiset.filter (fun a => Eq (f x) (f a)) s) b →  …
      -/
    · simp only [mem_filter, beq_iff_eq, and_imp, @eq_comm _ (f x), imp_self, implies_true]
      /-
        🎉 no goals
      -/


/-- `Multiset.map f` preserves `count` if `f` is injective -/
theorem count_map_eq_count' [DecidableEq β] (f : α → β) (s : Multiset α) (hf : Function.Injective f)
    (x : α) : (s.map f).count (f x) = s.count x := by
  /-
    α : Type u_1
    β : Type v
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    f : α → β
    s : Multiset α
    hf : Function.Injective f
    x : α
    ⊢ Eq (Multiset.count (f x) (Multiset.map f s)) (Multiset.count x s)
  -/
  by_cases H : x ∈ s
    /-
      case pos
      α : Type u_1
      β : Type v
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      f : α → β
      s : Multiset α
      hf : Function.Injective f
      x : α
      H : Membership.mem s x
      ⊢ Eq (Multiset.count (f x) (Multiset.map f s)) (Multiset.count x s)
    -/
  · exact count_map_eq_count f _ hf.injOn _ H
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type v
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      f : α → β
      s : Multiset α
      hf : Function.Injective f
      x : α
      H : Not (Membership.mem s x)
      ⊢ Eq (Multiset.count (f x) (Multiset.map f s)) (Multiset.count x s)
    -/
  · rw [count_eq_zero_of_not_mem H, count_eq_zero, mem_map]
    /-
      case neg
      α : Type u_1
      β : Type v
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      f : α → β
      s : Multiset α
      hf : Function.Injective f
      x : α
      H : Not (Membership.mem s x)
      ⊢ Not (Exists fun a => And (Membership.mem s a) (Eq (f a) (f x)))
    -/
    rintro ⟨k, hks, hkx⟩
    /-
      case neg.intro.intro
      α : Type u_1
      β : Type v
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      f : α → β
      s : Multiset α
      hf : Function.Injective f
      x : α
      H : Not (Membership.mem s x)
      k : α
      hks : Membership.mem s k
      hkx : Eq (f k) (f x)
      ⊢ False
    -/
    rw [hf hkx] at hks
    /-
      case neg.intro.intro
      α : Type u_1
      β : Type v
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      f : α → β
      s : Multiset α
      hf : Function.Injective f
      x : α
      H : Not (Membership.mem s x)
      k : α
      hks : Membership.mem s x
      hkx : Eq (f k) (f x)
      ⊢ False
    -/
    contradiction
    /-
      🎉 no goals
    -/


@[simp]
theorem sub_filter_eq_filter_not (p) [DecidablePred p] (s : Multiset α) :
    s - s.filter p = s.filter (fun a ↦ ¬ p a) := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    p : α → Prop
    inst✝ : DecidablePred p
    s : Multiset α
    ⊢ Eq (HSub.hSub s (Multiset.filter p s)) (Multiset.filter (fun a => Not (p a)) …
  -/
                              /-
                                🎉 no goals
                              -/
  ext a; by_cases h : p a <;> simp [h]
                              /-
                                🎉 no goals
                              -/


theorem filter_eq' (s : Multiset α) (b : α) : s.filter (· = b) = replicate (count b s) b :=
  Quotient.inductionOn s fun l => by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      s : Multiset α
      b : α
      l : List α
      ⊢ Eq (Multiset.filter (fun x => Eq x b) (Quotient.mk (List.isSetoid α) l)) (Mu …
    -/
    simp only [quot_mk_to_coe, filter_coe, mem_coe, coe_count]
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      s : Multiset α
      b : α
      l : List α
      ⊢ Eq (↑(List.filter (fun b_1 => Decidable.decide (Eq b_1 b)) l)) (Multiset.rep …
    -/
    rw [List.filter_eq l b, coe_replicate]
    /-
      🎉 no goals
    -/


theorem filter_eq (s : Multiset α) (b : α) : s.filter (Eq b) = replicate (count b s) b := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Multiset α
    b : α
    ⊢ Eq (Multiset.filter (Eq b) s) (Multiset.replicate (Multiset.count b s) b)
  -/
  simp_rw [← filter_eq', eq_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem replicate_inter (n : ℕ) (x : α) (s : Multiset α) :
    replicate n x ∩ s = replicate (min n (s.count x)) x := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    n : Nat
    x : α
    s : Multiset α
    ⊢ Eq (Inter.inter (Multiset.replicate n x) s) (Multiset.replicate (Min.min n ( …
  -/
  ext y
  /-
    case a
    α : Type u_1
    inst✝ : DecidableEq α
    n : Nat
    x : α
    s : Multiset α
    y : α
    ⊢ Eq (Multiset.count y (Inter.inter (Multiset.replicate n x) s)) (Multiset.cou …
  -/
  rw [count_inter, count_replicate, count_replicate]
  /-
    case a
    α : Type u_1
    inst✝ : DecidableEq α
    n : Nat
    x : α
    s : Multiset α
    y : α
    ⊢ Eq (Min.min (ite (Eq x y) n 0) (Multiset.count y s)) (ite (Eq x y) (Min.min  …
  -/
  by_cases h : x = y
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      n : Nat
      x : α
      s : Multiset α
      y : α
      h : Eq x y
      ⊢ Eq (Min.min (ite (Eq x y) n 0) (Multiset.count y s)) (ite (Eq x y) (Min.min  …
    -/
  · simp only [h, if_true]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      n : Nat
      x : α
      s : Multiset α
      y : α
      h : Not (Eq x y)
      ⊢ Eq (Min.min (ite (Eq x y) n 0) (Multiset.count y s)) (ite (Eq x y) (Min.min  …
    -/
  · simp only [h, if_false, Nat.zero_min]
    /-
      🎉 no goals
    -/


@[simp]
theorem inter_replicate (s : Multiset α) (n : ℕ) (x : α) :
    s ∩ replicate n x = replicate (min (s.count x) n) x := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Multiset α
    n : Nat
    x : α
    ⊢ Eq (Inter.inter s (Multiset.replicate n x)) (Multiset.replicate (Min.min (Mu …
  -/
  rw [inter_comm, replicate_inter, min_comm]
  /-
    🎉 no goals
  -/


theorem erase_attach_map_val (s : Multiset α) (x : {x // x ∈ s}) :
    (s.attach.erase x).map (↑) = s.erase x := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Multiset α
    x : Subtype fun x => Membership.mem s x
    ⊢ Eq (Multiset.map Subtype.val (s.attach.erase x)) (s.erase ↑x)
  -/
  rw [Multiset.map_erase _ val_injective, attach_map_val]
  /-
    🎉 no goals
  -/


theorem erase_attach_map (s : Multiset α) (f : α → β) (x : {x // x ∈ s}) :
    (s.attach.erase x).map (fun j : {x // x ∈ s} ↦ f j) = (s.erase x).map f := by
  /-
    α : Type u_1
    β : Type v
    inst✝ : DecidableEq α
    s : Multiset α
    f : α → β
    x : Subtype fun x => Membership.mem s x
    ⊢ Eq (Multiset.map (fun j => f ↑j) (s.attach.erase x)) (Multiset.map f (s.eras …
  -/
  simp only [← Function.comp_apply (f := f)]
  /-
    α : Type u_1
    β : Type v
    inst✝ : DecidableEq α
    s : Multiset α
    f : α → β
    x : Subtype fun x => Membership.mem s x
    ⊢ Eq (Multiset.map (fun x => Function.comp f Subtype.val x) (s.attach.erase x) …
  -/
  rw [← map_map, erase_attach_map_val]
  /-
    🎉 no goals
  -/


@[ext]
theorem addHom_ext [AddZeroClass β] ⦃f g : Multiset α →+ β⦄ (h : ∀ x, f {x} = g {x}) : f = g := by
  /-
    α : Type u_1
    β : Type v
    inst✝ : AddZeroClass β
    f g : AddMonoidHom (Multiset α) β
    h : ∀ (x : α), Eq (f (Singleton.singleton x)) (g (Singleton.singleton x))
    ⊢ Eq f g
  -/
  ext s
  /-
    case h
    α : Type u_1
    β : Type v
    inst✝ : AddZeroClass β
    f g : AddMonoidHom (Multiset α) β
    h : ∀ (x : α), Eq (f (Singleton.singleton x)) (g (Singleton.singleton x))
    s : Multiset α
    ⊢ Eq (f s) (g s)
  -/
  induction' s using Multiset.induction_on with a s ih
    /-
      case h.empty
      α : Type u_1
      β : Type v
      inst✝ : AddZeroClass β
      f g : AddMonoidHom (Multiset α) β
      h : ∀ (x : α), Eq (f (Singleton.singleton x)) (g (Singleton.singleton x))
      ⊢ Eq (f 0) (g 0)
    -/
  · simp only [_root_.map_zero]
    /-
      🎉 no goals
    -/
    /-
      case h.cons
      α : Type u_1
      β : Type v
      inst✝ : AddZeroClass β
      f g : AddMonoidHom (Multiset α) β
      h : ∀ (x : α), Eq (f (Singleton.singleton x)) (g (Singleton.singleton x))
      a : α
      s : Multiset α
      ih : Eq (f s) (g s)
      ⊢ Eq (f (Multiset.cons a s)) (g (Multiset.cons a s))
    -/
  · simp only [← singleton_add, _root_.map_add, ih, h]
    /-
      🎉 no goals
    -/


@[simp]
theorem map_le_map_iff {f : α → β} (hf : Function.Injective f) {s t : Multiset α} :
    s.map f ≤ t.map f ↔ s ≤ t := by
  classical
    refine ⟨fun h => le_iff_count.mpr fun a => ?_, map_le_map⟩
    simpa [count_map_eq_count' f _ hf] using le_iff_count.mp h (f a)


/-- Associate to an embedding `f` from `α` to `β` the order embedding that maps a multiset to its
image under `f`. -/
@[simps!]
def mapEmbedding (f : α ↪ β) : Multiset α ↪o Multiset β :=
  OrderEmbedding.ofMapLEIff (map f) fun _ _ => map_le_map_iff f.inj'


theorem count_eq_card_filter_eq [DecidableEq α] (s : Multiset α) (a : α) :
                                              /-
                                                α : Type u_1
                                                inst✝ : DecidableEq α
                                                s : Multiset α
                                                a : α
                                                ⊢ Eq (Multiset.count a s) (Multiset.filter (fun x => Eq a x) s).card
                                              -/
    s.count a = card (s.filter (a = ·)) := by rw [count, countP_eq_card_filter]
                                              /-
                                                🎉 no goals
                                              -/


/--
Mapping a multiset through a predicate and counting the `True`s yields the cardinality of the set
filtered by the predicate. Note that this uses the notion of a multiset of `Prop`s - due to the
decidability requirements of `count`, the decidability instance on the LHS is different from the
RHS. In particular, the decidability instance on the left leaks `Classical.decEq`.
See [here](https://github.com/leanprover-community/mathlib/pull/11306#discussion_r782286812)
for more discussion.
-/
@[simp]
theorem map_count_True_eq_filter_card (s : Multiset α) (p : α → Prop) [DecidablePred p] :
    (s.map p).count True = card (s.filter p) := by
  simp only [count_eq_card_filter_eq, filter_map, card_map, Function.id_comp,
    eq_true_eq_id, Function.comp_apply]


@[simp] theorem sub_singleton [DecidableEq α] (a : α) (s : Multiset α) : s - {a} = s.erase a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s : Multiset α
    ⊢ Eq (HSub.hSub s (Singleton.singleton a)) (s.erase a)
  -/
  ext
  /-
    case a
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s : Multiset α
    a✝ : α
    ⊢ Eq (Multiset.count a✝ (HSub.hSub s (Singleton.singleton a))) (Multiset.count …
  -/
  simp only [count_sub, count_singleton]
  /-
    case a
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s : Multiset α
    a✝ : α
    ⊢ Eq (HSub.hSub (Multiset.count a✝ s) (ite (Eq a✝ a) 1 0)) (Multiset.count a✝  …
  -/
            /-
              🎉 no goals
            -/
  split <;> simp_all
            /-
              🎉 no goals
            -/


theorem mem_sub [DecidableEq α] {a : α} {s t : Multiset α} :
    a ∈ s - t ↔ t.count a < s.count a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s t : Multiset α
    ⊢ Iff (Membership.mem (HSub.hSub s t) a) (LT.lt (Multiset.count a t) (Multiset …
  -/
  rw [← count_pos, count_sub, Nat.sub_pos_iff_lt]
  /-
    🎉 no goals
  -/


theorem inter_add_sub_of_add_eq_add [DecidableEq α] {M N P Q : Multiset α} (h : M + N = P + Q) :
    (N ∩ Q) + (P - M) = N := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    M N P Q : Multiset α
    h : Eq (HAdd.hAdd M N) (HAdd.hAdd P Q)
    ⊢ Eq (HAdd.hAdd (Inter.inter N Q) (HSub.hSub P M)) N
  -/
  ext x
  /-
    case a
    α : Type u_1
    inst✝ : DecidableEq α
    M N P Q : Multiset α
    h : Eq (HAdd.hAdd M N) (HAdd.hAdd P Q)
    x : α
    ⊢ Eq (Multiset.count x (HAdd.hAdd (Inter.inter N Q) (HSub.hSub P M))) (Multise …
  -/
  rw [Multiset.count_add, Multiset.count_inter, Multiset.count_sub]
  have h0 : M.count x + N.count x = P.count x + Q.count x := by
    rw [Multiset.ext] at h
    simp_all only [Multiset.mem_add, Multiset.count_add]
  /-
    case a
    α : Type u_1
    inst✝ : DecidableEq α
    M N P Q : Multiset α
    h : Eq (HAdd.hAdd M N) (HAdd.hAdd P Q)
    x : α
    h0 : Eq (HAdd.hAdd (Multiset.count x M) (Multiset.count x N)) (HAdd.hAdd (Mult …
    ⊢ Eq (HAdd.hAdd (Min.min (Multiset.count x N) (Multiset.count x Q)) (HSub.hSub …
  -/
  omega
  /-
    🎉 no goals
  -/


/-- `Rel r s t` -- lift the relation `r` between two elements to a relation between `s` and `t`,
s.t. there is a one-to-one mapping between elements in `s` and `t` following `r`. -/
@[mk_iff]
inductive Rel (r : α → β → Prop) : Multiset α → Multiset β → Prop
  | zero : Rel r 0 0
  | cons {a b as bs} : r a b → Rel r as bs → Rel r (a ::ₘ as) (b ::ₘ bs)


private theorem rel_flip_aux {s t} (h : Rel r s t) : Rel (flip r) t s :=
  Rel.recOn h Rel.zero fun h₀ _h₁ ih => Rel.cons h₀ ih


theorem rel_flip {s t} : Rel (flip r) s t ↔ Rel r t s :=
  ⟨rel_flip_aux, rel_flip_aux⟩


theorem rel_refl_of_refl_on {m : Multiset α} {r : α → α → Prop} : (∀ x ∈ m, r x x) → Rel r m m := by
  /-
    α : Type u_1
    m : Multiset α
    r : α → α → Prop
    ⊢ (∀ (x : α), Membership.mem m x → r x x) → Multiset.Rel r m m
  -/
  refine m.induction_on ?_ ?_
    /-
      case refine_1
      α : Type u_1
      m : Multiset α
      r : α → α → Prop
      ⊢ (∀ (x : α), Membership.mem 0 x → r x x) → Multiset.Rel r 0 0
    -/
  · intros
    /-
      case refine_1
      α : Type u_1
      m : Multiset α
      r : α → α → Prop
      a✝ : ∀ (x : α), Membership.mem 0 x → r x x
      ⊢ Multiset.Rel r 0 0
    -/
    apply Rel.zero
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m : Multiset α
      r : α → α → Prop
      ⊢ ∀ (a : α) (s : Multiset α), ((∀ (x : α), Membership.mem s x → r x x) → Multi …
    -/
  · intro a m ih h
    /-
      case refine_2
      α : Type u_1
      m✝ : Multiset α
      r : α → α → Prop
      a : α
      m : Multiset α
      ih : (∀ (x : α), Membership.mem m x → r x x) → Multiset.Rel r m m
      h : ∀ (x : α), Membership.mem (Multiset.cons a m) x → r x x
      ⊢ Multiset.Rel r (Multiset.cons a m) (Multiset.cons a m)
    -/
    exact Rel.cons (h _ (mem_cons_self _ _)) (ih fun _ ha => h _ (mem_cons_of_mem ha))
    /-
      🎉 no goals
    -/


theorem rel_eq_refl {s : Multiset α} : Rel (· = ·) s s :=
  rel_refl_of_refl_on fun _x _hx => rfl


theorem rel_eq {s t : Multiset α} : Rel (· = ·) s t ↔ s = t := by
  /-
    α : Type u_1
    s t : Multiset α
    ⊢ Iff (Multiset.Rel (fun x1 x2 => Eq x1 x2) s t) (Eq s t)
  -/
  constructor
    /-
      case mp
      α : Type u_1
      s t : Multiset α
      ⊢ Multiset.Rel (fun x1 x2 => Eq x1 x2) s t → Eq s t
    -/
  · intro h
    /-
      case mp
      α : Type u_1
      s t : Multiset α
      h : Multiset.Rel (fun x1 x2 => Eq x1 x2) s t
      ⊢ Eq s t
    -/
                    /-
                      🎉 no goals
                    -/
    induction h <;> simp [*]
                    /-
                      🎉 no goals
                    -/
    /-
      case mpr
      α : Type u_1
      s t : Multiset α
      ⊢ Eq s t → Multiset.Rel (fun x1 x2 => Eq x1 x2) s t
    -/
  · intro h
    /-
      case mpr
      α : Type u_1
      s t : Multiset α
      h : Eq s t
      ⊢ Multiset.Rel (fun x1 x2 => Eq x1 x2) s t
    -/
    subst h
    /-
      case mpr
      α : Type u_1
      s : Multiset α
      ⊢ Multiset.Rel (fun x1 x2 => Eq x1 x2) s s
    -/
    exact rel_eq_refl
    /-
      🎉 no goals
    -/


theorem Rel.mono {r p : α → β → Prop} {s t} (hst : Rel r s t)
    (h : ∀ a ∈ s, ∀ b ∈ t, r a b → p a b) : Rel p s t := by
  induction hst with
  | zero => exact Rel.zero
  | @cons a b s t hab _hst ih =>
    apply Rel.cons (h a (mem_cons_self _ _) b (mem_cons_self _ _) hab)
    exact ih fun a' ha' b' hb' h' => h a' (mem_cons_of_mem ha') b' (mem_cons_of_mem hb') h'


theorem Rel.add {s t u v} (hst : Rel r s t) (huv : Rel r u v) : Rel r (s + u) (t + v) := by
  induction hst with
  | zero => simpa using huv
  | cons hab hst ih => simpa using ih.cons hab


theorem rel_flip_eq {s t : Multiset α} : Rel (fun a b => b = a) s t ↔ s = t :=
                                         /-
                                           α : Type u_1
                                           s t : Multiset α
                                           ⊢ Iff (Multiset.Rel (flip fun x1 x2 => Eq x1 x2) s t) (Eq s t)
                                         -/
  show Rel (flip (· = ·)) s t ↔ s = t by rw [rel_flip, rel_eq, eq_comm]
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
                                                                 /-
                                                                   α : Type u_1
                                                                   β : Type v
                                                                   r : α → β → Prop
                                                                   b : Multiset β
                                                                   ⊢ Iff (Multiset.Rel r 0 b) (Eq b 0)
                                                                 -/
theorem rel_zero_left {b : Multiset β} : Rel r 0 b ↔ b = 0 := by rw [rel_iff]; simp
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


@[simp]
                                                                  /-
                                                                    α : Type u_1
                                                                    β : Type v
                                                                    r : α → β → Prop
                                                                    a : Multiset α
                                                                    ⊢ Iff (Multiset.Rel r a 0) (Eq a 0)
                                                                  -/
theorem rel_zero_right {a : Multiset α} : Rel r a 0 ↔ a = 0 := by rw [rel_iff]; simp
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


theorem rel_cons_left {a as bs} :
    Rel r (a ::ₘ as) bs ↔ ∃ b bs', r a b ∧ Rel r as bs' ∧ bs = b ::ₘ bs' := by
  /-
    α : Type u_1
    β : Type v
    r : α → β → Prop
    a : α
    as : Multiset α
    bs : Multiset β
    ⊢ Iff (Multiset.Rel r (Multiset.cons a as) bs) (Exists fun b => Exists fun bs' …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type v
      r : α → β → Prop
      a : α
      as : Multiset α
      bs : Multiset β
      ⊢ Multiset.Rel r (Multiset.cons a as) bs → Exists fun b => Exists fun bs' => A …
    -/
  · generalize hm : a ::ₘ as = m
    /-
      case mp
      α : Type u_1
      β : Type v
      r : α → β → Prop
      a : α
      as : Multiset α
      bs : Multiset β
      m : Multiset α
      hm : Eq (Multiset.cons a as) m
      ⊢ Multiset.Rel r m bs → Exists fun b => Exists fun bs' => And (r a b) (And (Mu …
    -/
    intro h
    induction h generalizing as with
    | zero => simp at hm
    | @cons a' b as' bs ha'b h ih =>
      rcases cons_eq_cons.1 hm with (⟨eq₁, eq₂⟩ | ⟨_h, cs, eq₁, eq₂⟩)
      · subst eq₁
        subst eq₂
        exact ⟨b, bs, ha'b, h, rfl⟩
      · rcases ih eq₂.symm with ⟨b', bs', h₁, h₂, eq⟩
        exact ⟨b', b ::ₘ bs', h₁, eq₁.symm ▸ Rel.cons ha'b h₂, eq.symm ▸ cons_swap _ _ _⟩
    /-
      case mpr
      α : Type u_1
      β : Type v
      r : α → β → Prop
      a : α
      as : Multiset α
      bs : Multiset β
      ⊢ (Exists fun b => Exists fun bs' => And (r a b) (And (Multiset.Rel r as bs')  …
    -/
  · exact fun ⟨b, bs', hab, h, Eq⟩ => Eq.symm ▸ Rel.cons hab h
    /-
      🎉 no goals
    -/


theorem rel_cons_right {as b bs} :
    Rel r as (b ::ₘ bs) ↔ ∃ a as', r a b ∧ Rel r as' bs ∧ as = a ::ₘ as' := by
  /-
    α : Type u_1
    β : Type v
    r : α → β → Prop
    as : Multiset α
    b : β
    bs : Multiset β
    ⊢ Iff (Multiset.Rel r as (Multiset.cons b bs)) (Exists fun a => Exists fun as' …
  -/
  rw [← rel_flip, rel_cons_left]
  /-
    α : Type u_1
    β : Type v
    r : α → β → Prop
    as : Multiset α
    b : β
    bs : Multiset β
    ⊢ Iff (Exists fun b_1 => Exists fun bs' => And (flip r b b_1) (And (Multiset.R …
  -/
  refine exists₂_congr fun a as' => ?_
  /-
    α : Type u_1
    β : Type v
    r : α → β → Prop
    as : Multiset α
    b : β
    bs : Multiset β
    a : α
    as' : Multiset α
    ⊢ Iff (And (flip r b a) (And (Multiset.Rel (flip r) bs as') (Eq as (Multiset.c …
  -/
  rw [rel_flip, flip]
  /-
    🎉 no goals
  -/


theorem rel_add_left {as₀ as₁} :
    ∀ {bs}, Rel r (as₀ + as₁) bs ↔ ∃ bs₀ bs₁, Rel r as₀ bs₀ ∧ Rel r as₁ bs₁ ∧ bs = bs₀ + bs₁ :=
                                  /-
                                    α : Type u_1
                                    β : Type v
                                    r : α → β → Prop
                                    as₀ as₁ : Multiset α
                                    ⊢ ∀ {bs : Multiset β}, Iff (Multiset.Rel r (HAdd.hAdd 0 as₁) bs) (Exists fun b …
                                  -/
  @(Multiset.induction_on as₀ (by simp) fun a s ih bs ↦ by
                                  /-
                                    🎉 no goals
                                  -/
      /-
        α : Type u_1
        β : Type v
        r : α → β → Prop
        as₀ as₁ : Multiset α
        a : α
        s : Multiset α
        ih : ∀ {bs : Multiset β}, Iff (Multiset.Rel r (HAdd.hAdd s as₁) bs) (Exists fu …
        bs : Multiset β
        ⊢ Iff (Multiset.Rel r (HAdd.hAdd (Multiset.cons a s) as₁) bs) (Exists fun bs₀  …
      -/
      simp only [ih, cons_add, rel_cons_left]
      /-
        α : Type u_1
        β : Type v
        r : α → β → Prop
        as₀ as₁ : Multiset α
        a : α
        s : Multiset α
        ih : ∀ {bs : Multiset β}, Iff (Multiset.Rel r (HAdd.hAdd s as₁) bs) (Exists fu …
        bs : Multiset β
        ⊢ Iff (Exists fun b => Exists fun bs' => And (r a b) (And (Exists fun bs₀ => E …
      -/
      constructor
        /-
          case mp
          α : Type u_1
          β : Type v
          r : α → β → Prop
          as₀ as₁ : Multiset α
          a : α
          s : Multiset α
          ih : ∀ {bs : Multiset β}, Iff (Multiset.Rel r (HAdd.hAdd s as₁) bs) (Exists fu …
          bs : Multiset β
          ⊢ (Exists fun b => Exists fun bs' => And (r a b) (And (Exists fun bs₀ => Exist …
        -/
      · intro h
        /-
          case mp
          α : Type u_1
          β : Type v
          r : α → β → Prop
          as₀ as₁ : Multiset α
          a : α
          s : Multiset α
          ih : ∀ {bs : Multiset β}, Iff (Multiset.Rel r (HAdd.hAdd s as₁) bs) (Exists fu …
          bs : Multiset β
          h : Exists fun b => Exists fun bs' => And (r a b) (And (Exists fun bs₀ => Exis …
          ⊢ Exists fun bs₀ => Exists fun bs₁ => And (Exists fun b => Exists fun bs' => A …
        -/
        rcases h with ⟨b, bs', hab, h, rfl⟩
        /-
          case mp.intro.intro.intro.intro
          α : Type u_1
          β : Type v
          r : α → β → Prop
          as₀ as₁ : Multiset α
          a : α
          s : Multiset α
          ih : ∀ {bs : Multiset β}, Iff (Multiset.Rel r (HAdd.hAdd s as₁) bs) (Exists fu …
          b : β
          bs' : Multiset β
          hab : r a b
          h : Exists fun bs₀ => Exists fun bs₁ => And (Multiset.Rel r s bs₀) (And (Multi …
          ⊢ Exists fun bs₀ => Exists fun bs₁ => And (Exists fun b => Exists fun bs' => A …
        -/
        rcases h with ⟨bs₀, bs₁, h₀, h₁, rfl⟩
        /-
          case mp.intro.intro.intro.intro.intro.intro.intro.intro
          α : Type u_1
          β : Type v
          r : α → β → Prop
          as₀ as₁ : Multiset α
          a : α
          s : Multiset α
          ih : ∀ {bs : Multiset β}, Iff (Multiset.Rel r (HAdd.hAdd s as₁) bs) (Exists fu …
          b : β
          hab : r a b
          bs₀ bs₁ : Multiset β
          h₀ : Multiset.Rel r s bs₀
          h₁ : Multiset.Rel r as₁ bs₁
          ⊢ Exists fun bs₀_1 => Exists fun bs₁_1 => And (Exists fun b => Exists fun bs'  …
        -/
        exact ⟨b ::ₘ bs₀, bs₁, ⟨b, bs₀, hab, h₀, rfl⟩, h₁, by simp⟩
        /-
          🎉 no goals
        -/
        /-
          case mpr
          α : Type u_1
          β : Type v
          r : α → β → Prop
          as₀ as₁ : Multiset α
          a : α
          s : Multiset α
          ih : ∀ {bs : Multiset β}, Iff (Multiset.Rel r (HAdd.hAdd s as₁) bs) (Exists fu …
          bs : Multiset β
          ⊢ (Exists fun bs₀ => Exists fun bs₁ => And (Exists fun b => Exists fun bs' =>  …
        -/
      · intro h
        /-
          case mpr
          α : Type u_1
          β : Type v
          r : α → β → Prop
          as₀ as₁ : Multiset α
          a : α
          s : Multiset α
          ih : ∀ {bs : Multiset β}, Iff (Multiset.Rel r (HAdd.hAdd s as₁) bs) (Exists fu …
          bs : Multiset β
          h : Exists fun bs₀ => Exists fun bs₁ => And (Exists fun b => Exists fun bs' => …
          ⊢ Exists fun b => Exists fun bs' => And (r a b) (And (Exists fun bs₀ => Exists …
        -/
        rcases h with ⟨bs₀, bs₁, h, h₁, rfl⟩
        /-
          case mpr.intro.intro.intro.intro
          α : Type u_1
          β : Type v
          r : α → β → Prop
          as₀ as₁ : Multiset α
          a : α
          s : Multiset α
          ih : ∀ {bs : Multiset β}, Iff (Multiset.Rel r (HAdd.hAdd s as₁) bs) (Exists fu …
          bs₀ bs₁ : Multiset β
          h : Exists fun b => Exists fun bs' => And (r a b) (And (Multiset.Rel r s bs')  …
          h₁ : Multiset.Rel r as₁ bs₁
          ⊢ Exists fun b => Exists fun bs' => And (r a b) (And (Exists fun bs₀ => Exists …
        -/
        rcases h with ⟨b, bs, hab, h₀, rfl⟩
        /-
          case mpr.intro.intro.intro.intro.intro.intro.intro.intro
          α : Type u_1
          β : Type v
          r : α → β → Prop
          as₀ as₁ : Multiset α
          a : α
          s : Multiset α
          ih : ∀ {bs : Multiset β}, Iff (Multiset.Rel r (HAdd.hAdd s as₁) bs) (Exists fu …
          bs₁ : Multiset β
          h₁ : Multiset.Rel r as₁ bs₁
          b : β
          bs : Multiset β
          hab : r a b
          h₀ : Multiset.Rel r s bs
          ⊢ Exists fun b_1 => Exists fun bs' => And (r a b_1) (And (Exists fun bs₀ => Ex …
        -/
        exact ⟨b, bs + bs₁, hab, ⟨bs, bs₁, h₀, h₁, rfl⟩, by simp⟩)
        /-
          🎉 no goals
        -/


theorem rel_add_right {as bs₀ bs₁} :
    Rel r as (bs₀ + bs₁) ↔ ∃ as₀ as₁, Rel r as₀ bs₀ ∧ Rel r as₁ bs₁ ∧ as = as₀ + as₁ := by
  /-
    α : Type u_1
    β : Type v
    r : α → β → Prop
    as : Multiset α
    bs₀ bs₁ : Multiset β
    ⊢ Iff (Multiset.Rel r as (HAdd.hAdd bs₀ bs₁)) (Exists fun as₀ => Exists fun as …
  -/
  rw [← rel_flip, rel_add_left]; simp [rel_flip]
                                 /-
                                   🎉 no goals
                                 -/


theorem rel_map_left {s : Multiset γ} {f : γ → α} :
    ∀ {t}, Rel r (s.map f) t ↔ Rel (fun a b => r (f a) b) s t :=
                                /-
                                  α : Type u_1
                                  β : Type v
                                  γ : Type u_2
                                  r : α → β → Prop
                                  s : Multiset γ
                                  f : γ → α
                                  ⊢ ∀ {t : Multiset β}, Iff (Multiset.Rel r (Multiset.map f 0) t) (Multiset.Rel  …
                                -/
                                /-
                                  🎉 no goals
                                -/
  @(Multiset.induction_on s (by simp) (by simp +contextual [rel_cons_left]))
                                          /-
                                            🎉 no goals
                                          -/


theorem rel_map_right {s : Multiset α} {t : Multiset γ} {f : γ → β} :
    Rel r s (t.map f) ↔ Rel (fun a b => r a (f b)) s t := by
  /-
    α : Type u_1
    β : Type v
    γ : Type u_2
    r : α → β → Prop
    s : Multiset α
    t : Multiset γ
    f : γ → β
    ⊢ Iff (Multiset.Rel r s (Multiset.map f t)) (Multiset.Rel (fun a b => r a (f b …
  -/
  rw [← rel_flip, rel_map_left, ← rel_flip]; rfl
                                             /-
                                               🎉 no goals
                                             -/


theorem rel_map {s : Multiset α} {t : Multiset β} {f : α → γ} {g : β → δ} :
    Rel p (s.map f) (t.map g) ↔ Rel (fun a b => p (f a) (g b)) s t :=
  rel_map_left.trans rel_map_right


theorem card_eq_card_of_rel {r : α → β → Prop} {s : Multiset α} {t : Multiset β} (h : Rel r s t) :
                          /-
                            α : Type u_1
                            β : Type v
                            r : α → β → Prop
                            s : Multiset α
                            t : Multiset β
                            h : Multiset.Rel r s t
                            ⊢ Eq s.card t.card
                          -/
                                          /-
                                            🎉 no goals
                                          -/
    card s = card t := by induction h <;> simp [*]
                                          /-
                                            🎉 no goals
                                          -/


theorem exists_mem_of_rel_of_mem {r : α → β → Prop} {s : Multiset α} {t : Multiset β}
    (h : Rel r s t) : ∀ {a : α}, a ∈ s → ∃ b ∈ t, r a b := by
  /-
    α : Type u_1
    β : Type v
    r : α → β → Prop
    s : Multiset α
    t : Multiset β
    h : Multiset.Rel r s t
    ⊢ ∀ {a : α}, Membership.mem s a → Exists fun b => And (Membership.mem t b) (r  …
  -/
  induction' h with x y s t hxy _hst ih
    /-
      case zero
      α : Type u_1
      β : Type v
      r : α → β → Prop
      s : Multiset α
      t : Multiset β
      ⊢ ∀ {a : α}, Membership.mem 0 a → Exists fun b => And (Membership.mem 0 b) (r  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      β : Type v
      r : α → β → Prop
      s✝ : Multiset α
      t✝ : Multiset β
      x : α
      y : β
      s : Multiset α
      t : Multiset β
      hxy : r x y
      _hst : Multiset.Rel r s t
      ih : ∀ {a : α}, Membership.mem s a → Exists fun b => And (Membership.mem t b)  …
      ⊢ ∀ {a : α}, Membership.mem (Multiset.cons x s) a → Exists fun b => And (Membe …
    -/
  · intro a ha
    /-
      case cons
      α : Type u_1
      β : Type v
      r : α → β → Prop
      s✝ : Multiset α
      t✝ : Multiset β
      x : α
      y : β
      s : Multiset α
      t : Multiset β
      hxy : r x y
      _hst : Multiset.Rel r s t
      ih : ∀ {a : α}, Membership.mem s a → Exists fun b => And (Membership.mem t b)  …
      a : α
      ha : Membership.mem (Multiset.cons x s) a
      ⊢ Exists fun b => And (Membership.mem (Multiset.cons y t) b) (r a b)
    -/
    cases' mem_cons.1 ha with ha ha
      /-
        case cons.inl
        α : Type u_1
        β : Type v
        r : α → β → Prop
        s✝ : Multiset α
        t✝ : Multiset β
        x : α
        y : β
        s : Multiset α
        t : Multiset β
        hxy : r x y
        _hst : Multiset.Rel r s t
        ih : ∀ {a : α}, Membership.mem s a → Exists fun b => And (Membership.mem t b)  …
        a : α
        ha✝ : Membership.mem (Multiset.cons x s) a
        ha : Eq a x
        ⊢ Exists fun b => And (Membership.mem (Multiset.cons y t) b) (r a b)
      -/
    · exact ⟨y, mem_cons_self _ _, ha.symm ▸ hxy⟩
      /-
        🎉 no goals
      -/
      /-
        case cons.inr
        α : Type u_1
        β : Type v
        r : α → β → Prop
        s✝ : Multiset α
        t✝ : Multiset β
        x : α
        y : β
        s : Multiset α
        t : Multiset β
        hxy : r x y
        _hst : Multiset.Rel r s t
        ih : ∀ {a : α}, Membership.mem s a → Exists fun b => And (Membership.mem t b)  …
        a : α
        ha✝ : Membership.mem (Multiset.cons x s) a
        ha : Membership.mem s a
        ⊢ Exists fun b => And (Membership.mem (Multiset.cons y t) b) (r a b)
      -/
    · rcases ih ha with ⟨b, hbt, hab⟩
      /-
        case cons.inr.intro.intro
        α : Type u_1
        β : Type v
        r : α → β → Prop
        s✝ : Multiset α
        t✝ : Multiset β
        x : α
        y : β
        s : Multiset α
        t : Multiset β
        hxy : r x y
        _hst : Multiset.Rel r s t
        ih : ∀ {a : α}, Membership.mem s a → Exists fun b => And (Membership.mem t b)  …
        a : α
        ha✝ : Membership.mem (Multiset.cons x s) a
        ha : Membership.mem s a
        b : β
        hbt : Membership.mem t b
        hab : r a b
        ⊢ Exists fun b => And (Membership.mem (Multiset.cons y t) b) (r a b)
      -/
      exact ⟨b, mem_cons.2 (Or.inr hbt), hab⟩
      /-
        🎉 no goals
      -/


theorem rel_of_forall {m1 m2 : Multiset α} {r : α → α → Prop} (h : ∀ a b, a ∈ m1 → b ∈ m2 → r a b)
    (hc : card m1 = card m2) : m1.Rel r m2 := by
  /-
    α : Type u_1
    m1 m2 : Multiset α
    r : α → α → Prop
    h : ∀ (a b : α), Membership.mem m1 a → Membership.mem m2 b → r a b
    hc : Eq m1.card m2.card
    ⊢ Multiset.Rel r m1 m2
  -/
  revert m1
  /-
    α : Type u_1
    m2 : Multiset α
    r : α → α → Prop
    ⊢ ∀ {m1 : Multiset α}, (∀ (a b : α), Membership.mem m1 a → Membership.mem m2 b …
  -/
  refine @(m2.induction_on ?_ ?_)
    /-
      case refine_1
      α : Type u_1
      m2 : Multiset α
      r : α → α → Prop
      ⊢ ∀ {m1 : Multiset α}, (∀ (a b : α), Membership.mem m1 a → Membership.mem 0 b  …
    -/
  · intro m _h hc
    /-
      case refine_1
      α : Type u_1
      m2 : Multiset α
      r : α → α → Prop
      m : Multiset α
      _h : ∀ (a b : α), Membership.mem m a → Membership.mem 0 b → r a b
      hc : Eq m.card (Multiset.card 0)
      ⊢ Multiset.Rel r m 0
    -/
    rw [rel_zero_right, ← card_eq_zero, hc, card_zero]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m2 : Multiset α
      r : α → α → Prop
      ⊢ ∀ (a : α) (s : Multiset α), (∀ {m1 : Multiset α}, (∀ (a b : α), Membership.m …
    -/
  · intro a t ih m h hc
    /-
      case refine_2
      α : Type u_1
      m2 : Multiset α
      r : α → α → Prop
      a : α
      t : Multiset α
      ih : ∀ {m1 : Multiset α}, (∀ (a b : α), Membership.mem m1 a → Membership.mem t …
      m : Multiset α
      h : ∀ (a_1 b : α), Membership.mem m a_1 → Membership.mem (Multiset.cons a t) b …
      hc : Eq m.card (Multiset.cons a t).card
      ⊢ Multiset.Rel r m (Multiset.cons a t)
    -/
    rw [card_cons] at hc
    /-
      case refine_2
      α : Type u_1
      m2 : Multiset α
      r : α → α → Prop
      a : α
      t : Multiset α
      ih : ∀ {m1 : Multiset α}, (∀ (a b : α), Membership.mem m1 a → Membership.mem t …
      m : Multiset α
      h : ∀ (a_1 b : α), Membership.mem m a_1 → Membership.mem (Multiset.cons a t) b …
      hc : Eq m.card (HAdd.hAdd t.card 1)
      ⊢ Multiset.Rel r m (Multiset.cons a t)
    -/
    obtain ⟨b, hb⟩ := card_pos_iff_exists_mem.1 (show 0 < card m from hc.symm ▸ Nat.succ_pos _)
    /-
      case refine_2.intro
      α : Type u_1
      m2 : Multiset α
      r : α → α → Prop
      a : α
      t : Multiset α
      ih : ∀ {m1 : Multiset α}, (∀ (a b : α), Membership.mem m1 a → Membership.mem t …
      m : Multiset α
      h : ∀ (a_1 b : α), Membership.mem m a_1 → Membership.mem (Multiset.cons a t) b …
      hc : Eq m.card (HAdd.hAdd t.card 1)
      b : α
      hb : Membership.mem m b
      ⊢ Multiset.Rel r m (Multiset.cons a t)
    -/
    obtain ⟨m', rfl⟩ := exists_cons_of_mem hb
    /-
      case refine_2.intro.intro
      α : Type u_1
      m2 : Multiset α
      r : α → α → Prop
      a : α
      t : Multiset α
      ih : ∀ {m1 : Multiset α}, (∀ (a b : α), Membership.mem m1 a → Membership.mem t …
      b : α
      m' : Multiset α
      h : ∀ (a_1 b_1 : α), Membership.mem (Multiset.cons b m') a_1 → Membership.mem  …
      hc : Eq (Multiset.cons b m').card (HAdd.hAdd t.card 1)
      hb : Membership.mem (Multiset.cons b m') b
      ⊢ Multiset.Rel r (Multiset.cons b m') (Multiset.cons a t)
    -/
    refine rel_cons_right.mpr ⟨b, m', h _ _ hb (mem_cons_self _ _), ih ?_ ?_, rfl⟩
      /-
        case refine_2.intro.intro.refine_1
        α : Type u_1
        m2 : Multiset α
        r : α → α → Prop
        a : α
        t : Multiset α
        ih : ∀ {m1 : Multiset α}, (∀ (a b : α), Membership.mem m1 a → Membership.mem t …
        b : α
        m' : Multiset α
        h : ∀ (a_1 b_1 : α), Membership.mem (Multiset.cons b m') a_1 → Membership.mem  …
        hc : Eq (Multiset.cons b m').card (HAdd.hAdd t.card 1)
        hb : Membership.mem (Multiset.cons b m') b
        ⊢ ∀ (a b : α), Membership.mem m' a → Membership.mem t b → r a b
      -/
    · exact fun _ _ ha hb => h _ _ (mem_cons_of_mem ha) (mem_cons_of_mem hb)
      /-
        🎉 no goals
      -/
      /-
        case refine_2.intro.intro.refine_2
        α : Type u_1
        m2 : Multiset α
        r : α → α → Prop
        a : α
        t : Multiset α
        ih : ∀ {m1 : Multiset α}, (∀ (a b : α), Membership.mem m1 a → Membership.mem t …
        b : α
        m' : Multiset α
        h : ∀ (a_1 b_1 : α), Membership.mem (Multiset.cons b m') a_1 → Membership.mem  …
        hc : Eq (Multiset.cons b m').card (HAdd.hAdd t.card 1)
        hb : Membership.mem (Multiset.cons b m') b
        ⊢ Eq m'.card t.card
      -/
    · simpa using hc
      /-
        🎉 no goals
      -/


theorem rel_replicate_left {m : Multiset α} {a : α} {r : α → α → Prop} {n : ℕ} :
    (replicate n a).Rel r m ↔ card m = n ∧ ∀ x, x ∈ m → r a x :=
  ⟨fun h =>
    ⟨(card_eq_card_of_rel h).symm.trans (card_replicate _ _), fun x hx => by
      /-
        α : Type u_1
        m : Multiset α
        a : α
        r : α → α → Prop
        n : Nat
        h : Multiset.Rel r (Multiset.replicate n a) m
        x : α
        hx : Membership.mem m x
        ⊢ r a x
      -/
      obtain ⟨b, hb1, hb2⟩ := exists_mem_of_rel_of_mem (rel_flip.2 h) hx
      /-
        case intro.intro
        α : Type u_1
        m : Multiset α
        a : α
        r : α → α → Prop
        n : Nat
        h : Multiset.Rel r (Multiset.replicate n a) m
        x : α
        hx : Membership.mem m x
        b : α
        hb1 : Membership.mem (Multiset.replicate n a) b
        hb2 : flip r x b
        ⊢ r a x
      -/
      rwa [eq_of_mem_replicate hb1] at hb2⟩,
      /-
        🎉 no goals
      -/
    fun h =>
    rel_of_forall (fun _ _ hx hy => (eq_of_mem_replicate hx).symm ▸ h.2 _ hy)
      (Eq.trans (card_replicate _ _) h.1.symm)⟩


theorem rel_replicate_right {m : Multiset α} {a : α} {r : α → α → Prop} {n : ℕ} :
    m.Rel r (replicate n a) ↔ card m = n ∧ ∀ x, x ∈ m → r x a :=
  rel_flip.trans rel_replicate_left


protected nonrec -- Porting note: added
theorem Rel.trans (r : α → α → Prop) [IsTrans α r] {s t u : Multiset α} (r1 : Rel r s t)
    (r2 : Rel r t u) : Rel r s u := by
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsTrans α r
    s t u : Multiset α
    r1 : Multiset.Rel r s t
    r2 : Multiset.Rel r t u
    ⊢ Multiset.Rel r s u
  -/
  induction' t using Multiset.induction_on with x t ih generalizing s u
    /-
      case empty
      α : Type u_1
      r : α → α → Prop
      inst✝ : IsTrans α r
      s u : Multiset α
      r1 : Multiset.Rel r s 0
      r2 : Multiset.Rel r 0 u
      ⊢ Multiset.Rel r s u
    -/
  · rw [rel_zero_right.mp r1, rel_zero_left.mp r2, rel_zero_left]
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      r : α → α → Prop
      inst✝ : IsTrans α r
      x : α
      t : Multiset α
      ih : ∀ {s u : Multiset α}, Multiset.Rel r s t → Multiset.Rel r t u → Multiset. …
      s u : Multiset α
      r1 : Multiset.Rel r s (Multiset.cons x t)
      r2 : Multiset.Rel r (Multiset.cons x t) u
      ⊢ Multiset.Rel r s u
    -/
  · obtain ⟨a, as, ha1, ha2, rfl⟩ := rel_cons_right.mp r1
    /-
      case cons.intro.intro.intro.intro
      α : Type u_1
      r : α → α → Prop
      inst✝ : IsTrans α r
      x : α
      t : Multiset α
      ih : ∀ {s u : Multiset α}, Multiset.Rel r s t → Multiset.Rel r t u → Multiset. …
      u : Multiset α
      r2 : Multiset.Rel r (Multiset.cons x t) u
      a : α
      as : Multiset α
      ha1 : r a x
      ha2 : Multiset.Rel r as t
      r1 : Multiset.Rel r (Multiset.cons a as) (Multiset.cons x t)
      ⊢ Multiset.Rel r (Multiset.cons a as) u
    -/
    obtain ⟨b, bs, hb1, hb2, rfl⟩ := rel_cons_left.mp r2
    /-
      case cons.intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      r : α → α → Prop
      inst✝ : IsTrans α r
      x : α
      t : Multiset α
      ih : ∀ {s u : Multiset α}, Multiset.Rel r s t → Multiset.Rel r t u → Multiset. …
      a : α
      as : Multiset α
      ha1 : r a x
      ha2 : Multiset.Rel r as t
      r1 : Multiset.Rel r (Multiset.cons a as) (Multiset.cons x t)
      b : α
      bs : Multiset α
      hb1 : r x b
      hb2 : Multiset.Rel r t bs
      r2 : Multiset.Rel r (Multiset.cons x t) (Multiset.cons b bs)
      ⊢ Multiset.Rel r (Multiset.cons a as) (Multiset.cons b bs)
    -/
    exact Multiset.Rel.cons (_root_.trans ha1 hb1) (ih ha2 hb2)
    /-
      🎉 no goals
    -/


theorem Rel.countP_eq (r : α → α → Prop) [IsTrans α r] [IsSymm α r] {s t : Multiset α} (x : α)
    [DecidablePred (r x)] (h : Rel r s t) : countP (r x) s = countP (r x) t := by
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝² : IsTrans α r
    inst✝¹ : IsSymm α r
    s t : Multiset α
    x : α
    inst✝ : DecidablePred (r x)
    h : Multiset.Rel r s t
    ⊢ Eq (Multiset.countP (r x) s) (Multiset.countP (r x) t)
  -/
  induction' s using Multiset.induction_on with y s ih generalizing t
    /-
      case empty
      α : Type u_1
      r : α → α → Prop
      inst✝² : IsTrans α r
      inst✝¹ : IsSymm α r
      x : α
      inst✝ : DecidablePred (r x)
      t : Multiset α
      h : Multiset.Rel r 0 t
      ⊢ Eq (Multiset.countP (r x) 0) (Multiset.countP (r x) t)
    -/
  · rw [rel_zero_left.mp h]
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      r : α → α → Prop
      inst✝² : IsTrans α r
      inst✝¹ : IsSymm α r
      x : α
      inst✝ : DecidablePred (r x)
      y : α
      s : Multiset α
      ih : ∀ {t : Multiset α}, Multiset.Rel r s t → Eq (Multiset.countP (r x) s) (Mu …
      t : Multiset α
      h : Multiset.Rel r (Multiset.cons y s) t
      ⊢ Eq (Multiset.countP (r x) (Multiset.cons y s)) (Multiset.countP (r x) t)
    -/
  · obtain ⟨b, bs, hb1, hb2, rfl⟩ := rel_cons_left.mp h
    /-
      case cons.intro.intro.intro.intro
      α : Type u_1
      r : α → α → Prop
      inst✝² : IsTrans α r
      inst✝¹ : IsSymm α r
      x : α
      inst✝ : DecidablePred (r x)
      y : α
      s : Multiset α
      ih : ∀ {t : Multiset α}, Multiset.Rel r s t → Eq (Multiset.countP (r x) s) (Mu …
      b : α
      bs : Multiset α
      hb1 : r y b
      hb2 : Multiset.Rel r s bs
      h : Multiset.Rel r (Multiset.cons y s) (Multiset.cons b bs)
      ⊢ Eq (Multiset.countP (r x) (Multiset.cons y s)) (Multiset.countP (r x) (Multi …
    -/
    rw [countP_cons, countP_cons, ih hb2]
    /-
      case cons.intro.intro.intro.intro
      α : Type u_1
      r : α → α → Prop
      inst✝² : IsTrans α r
      inst✝¹ : IsSymm α r
      x : α
      inst✝ : DecidablePred (r x)
      y : α
      s : Multiset α
      ih : ∀ {t : Multiset α}, Multiset.Rel r s t → Eq (Multiset.countP (r x) s) (Mu …
      b : α
      bs : Multiset α
      hb1 : r y b
      hb2 : Multiset.Rel r s bs
      h : Multiset.Rel r (Multiset.cons y s) (Multiset.cons b bs)
      ⊢ Eq (HAdd.hAdd (Multiset.countP (r x) bs) (ite (r x y) 1 0)) (HAdd.hAdd (Mult …
    -/
    simp only [decide_eq_true_eq, Nat.add_right_inj]
    /-
      case cons.intro.intro.intro.intro
      α : Type u_1
      r : α → α → Prop
      inst✝² : IsTrans α r
      inst✝¹ : IsSymm α r
      x : α
      inst✝ : DecidablePred (r x)
      y : α
      s : Multiset α
      ih : ∀ {t : Multiset α}, Multiset.Rel r s t → Eq (Multiset.countP (r x) s) (Mu …
      b : α
      bs : Multiset α
      hb1 : r y b
      hb2 : Multiset.Rel r s bs
      h : Multiset.Rel r (Multiset.cons y s) (Multiset.cons b bs)
      ⊢ Eq (ite (r x y) 1 0) (ite (r x b) 1 0)
    -/
    exact (if_congr ⟨fun h => _root_.trans h hb1, fun h => _root_.trans h (symm hb1)⟩ rfl rfl)
    /-
      🎉 no goals
    -/


theorem map_eq_map {f : α → β} (hf : Function.Injective f) {s t : Multiset α} :
    s.map f = t.map f ↔ s = t := by
  /-
    α : Type u_1
    β : Type v
    f : α → β
    hf : Function.Injective f
    s t : Multiset α
    ⊢ Iff (Eq (Multiset.map f s) (Multiset.map f t)) (Eq s t)
  -/
  rw [← rel_eq, ← rel_eq, rel_map]
  /-
    α : Type u_1
    β : Type v
    f : α → β
    hf : Function.Injective f
    s t : Multiset α
    ⊢ Iff (Multiset.Rel (fun a b => Eq (f a) (f b)) s t) (Multiset.Rel (fun x1 x2  …
  -/
  simp only [hf.eq_iff]
  /-
    🎉 no goals
  -/


theorem map_injective {f : α → β} (hf : Function.Injective f) :
    Function.Injective (Multiset.map f) := fun _x _y => (map_eq_map hf).1


lemma filter_attach' (s : Multiset α) (p : {a // a ∈ s} → Prop) [DecidableEq α]
    [DecidablePred p] :
    s.attach.filter p =
      (s.filter fun x ↦ ∃ h, p ⟨x, h⟩).attach.map (Subtype.map id fun _ ↦ mem_of_mem_filter) := by
  classical
  refine Multiset.map_injective Subtype.val_injective ?_
  rw [map_filter' _ Subtype.val_injective]
  simp only [Function.comp, Subtype.exists, coe_mk, Subtype.map,
    exists_and_right, exists_eq_right, attach_map_val, map_map, map_coe, id]


theorem map_mk_eq_map_mk_of_rel {r : α → α → Prop} {s t : Multiset α} (hst : s.Rel r t) :
    s.map (Quot.mk r) = t.map (Quot.mk r) :=
                                          /-
                                            α : Type u_1
                                            r : α → α → Prop
                                            s t : Multiset α
                                            hst : Multiset.Rel r s t
                                            a✝ b✝ : α
                                            as✝ bs✝ : Multiset α
                                            hab : r a✝ b✝
                                            _hst : Multiset.Rel r as✝ bs✝
                                            ih : Eq (Multiset.map (Quot.mk r) as✝) (Multiset.map (Quot.mk r) bs✝)
                                            ⊢ Eq (Multiset.map (Quot.mk r) (Multiset.cons a✝ as✝)) (Multiset.map (Quot.mk  …
                                          -/
  Rel.recOn hst rfl fun hab _hst ih => by simp [ih, Quot.sound hab]
                                          /-
                                            🎉 no goals
                                          -/


theorem exists_multiset_eq_map_quot_mk {r : α → α → Prop} (s : Multiset (Quot r)) :
    ∃ t : Multiset α, s = t.map (Quot.mk r) :=
  Multiset.induction_on s ⟨0, rfl⟩ fun a _s ⟨t, ht⟩ =>
    Quot.inductionOn a fun a => ht.symm ▸ ⟨a ::ₘ t, (map_cons _ _ _).symm⟩


theorem induction_on_multiset_quot {r : α → α → Prop} {p : Multiset (Quot r) → Prop}
    (s : Multiset (Quot r)) : (∀ s : Multiset α, p (s.map (Quot.mk r))) → p s :=
  match s, exists_multiset_eq_map_quot_mk s with
  | _, ⟨_t, rfl⟩ => fun h => h _


/-- `Disjoint s t` means that `s` and `t` have no elements in common. -/
@[deprecated _root_.Disjoint (since := "2024-11-01")]
protected def Disjoint (s t : Multiset α) : Prop :=
  ∀ ⦃a⦄, a ∈ s → a ∈ t → False


theorem disjoint_left {s t : Multiset α} : Disjoint s t ↔ ∀ {a}, a ∈ s → a ∉ t := by
  /-
    α : Type u_1
    s t : Multiset α
    ⊢ Iff (Disjoint s t) (∀ {a : α}, Membership.mem s a → Not (Membership.mem t a))
  -/
  refine ⟨fun h a hs ht ↦ ?_, fun h u hs ht ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      s t : Multiset α
      h : Disjoint s t
      a : α
      hs : Membership.mem s a
      ht : Membership.mem t a
      ⊢ False
    -/
  · simpa using h (singleton_le.mpr hs) (singleton_le.mpr ht)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      s t : Multiset α
      h : ∀ {a : α}, Membership.mem s a → Not (Membership.mem t a)
      u : Multiset α
      hs : LE.le u s
      ht : LE.le u t
      ⊢ LE.le u Bot.bot
    -/
  · rw [le_bot_iff, bot_eq_zero, eq_zero_iff_forall_not_mem]
    /-
      case refine_2
      α : Type u_1
      s t : Multiset α
      h : ∀ {a : α}, Membership.mem s a → Not (Membership.mem t a)
      u : Multiset α
      hs : LE.le u s
      ht : LE.le u t
      ⊢ ∀ (a : α), Not (Membership.mem u a)
    -/
    exact fun a ha ↦ h (subset_of_le hs ha) (subset_of_le ht ha)
    /-
      🎉 no goals
    -/


@[simp, norm_cast]
theorem coe_disjoint (l₁ l₂ : List α) : Disjoint (l₁ : Multiset α) l₂ ↔ l₁.Disjoint l₂ :=
  disjoint_left


@[deprecated (since := "2024-11-01")] protected alias Disjoint.symm := _root_.Disjoint.symm

@[deprecated (since := "2024-11-01")] protected alias disjoint_comm := _root_.disjoint_comm


theorem disjoint_right {s t : Multiset α} : Disjoint s t ↔ ∀ {a}, a ∈ t → a ∉ s :=
  disjoint_comm.trans disjoint_left


theorem disjoint_iff_ne {s t : Multiset α} : Disjoint s t ↔ ∀ a ∈ s, ∀ b ∈ t, a ≠ b := by
  /-
    α : Type u_1
    s t : Multiset α
    ⊢ Iff (Disjoint s t) (∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.me …
  -/
  simp [disjoint_left, imp_not_comm]
  /-
    🎉 no goals
  -/


theorem disjoint_of_subset_left {s t u : Multiset α} (h : s ⊆ u) (d : Disjoint u t) :
    Disjoint s t :=
  disjoint_left.mpr fun ha ↦ disjoint_left.mp d <| h ha


theorem disjoint_of_subset_right {s t u : Multiset α} (h : t ⊆ u) (d : Disjoint s u) :
    Disjoint s t :=
  (disjoint_of_subset_left h d.symm).symm


@[deprecated (since := "2024-11-01")] protected alias disjoint_of_le_left := Disjoint.mono_left

@[deprecated (since := "2024-11-01")] protected alias disjoint_of_le_right := Disjoint.mono_right


@[simp]
theorem zero_disjoint (l : Multiset α) : Disjoint 0 l := disjoint_bot_left


@[simp]
theorem singleton_disjoint {l : Multiset α} {a : α} : Disjoint {a} l ↔ a ∉ l := by
  /-
    α : Type u_1
    l : Multiset α
    a : α
    ⊢ Iff (Disjoint (Singleton.singleton a) l) (Not (Membership.mem l a))
  -/
  simp [disjoint_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem disjoint_singleton {l : Multiset α} {a : α} : Disjoint l {a} ↔ a ∉ l := by
  /-
    α : Type u_1
    l : Multiset α
    a : α
    ⊢ Iff (Disjoint l (Singleton.singleton a)) (Not (Membership.mem l a))
  -/
  rw [_root_.disjoint_comm, singleton_disjoint]
  /-
    🎉 no goals
  -/


@[simp]
theorem disjoint_add_left {s t u : Multiset α} :
                                                           /-
                                                             α : Type u_1
                                                             s t u : Multiset α
                                                             ⊢ Iff (Disjoint (HAdd.hAdd s t) u) (And (Disjoint s u) (Disjoint t u))
                                                           -/
    Disjoint (s + t) u ↔ Disjoint s u ∧ Disjoint t u := by simp [disjoint_left, or_imp, forall_and]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
theorem disjoint_add_right {s t u : Multiset α} :
    Disjoint s (t + u) ↔ Disjoint s t ∧ Disjoint s u := by
  /-
    α : Type u_1
    s t u : Multiset α
    ⊢ Iff (Disjoint s (HAdd.hAdd t u)) (And (Disjoint s t) (Disjoint s u))
  -/
  rw [_root_.disjoint_comm, disjoint_add_left]; tauto
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
theorem disjoint_cons_left {a : α} {s t : Multiset α} :
    Disjoint (a ::ₘ s) t ↔ a ∉ t ∧ Disjoint s t :=
                                             /-
                                               α : Type u_1
                                               a : α
                                               s t : Multiset α
                                               ⊢ Iff (And (Disjoint (Singleton.singleton a) t) (Disjoint s t)) (And (Not (Mem …
                                             -/
  (@disjoint_add_left _ {a} s t).trans <| by rw [singleton_disjoint]
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
theorem disjoint_cons_right {a : α} {s t : Multiset α} :
    Disjoint s (a ::ₘ t) ↔ a ∉ s ∧ Disjoint s t := by
  /-
    α : Type u_1
    a : α
    s t : Multiset α
    ⊢ Iff (Disjoint s (Multiset.cons a t)) (And (Not (Membership.mem s a)) (Disjoi …
  -/
  rw [_root_.disjoint_comm, disjoint_cons_left]; tauto
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem inter_eq_zero_iff_disjoint [DecidableEq α] {s t : Multiset α} :
                                   /-
                                     α : Type u_1
                                     inst✝ : DecidableEq α
                                     s t : Multiset α
                                     ⊢ Iff (Eq (Inter.inter s t) 0) (Disjoint s t)
                                   -/
    s ∩ t = 0 ↔ Disjoint s t := by rw [← subset_zero]; simp [subset_iff, disjoint_left]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
theorem disjoint_union_left [DecidableEq α] {s t u : Multiset α} :
    Disjoint (s ∪ t) u ↔ Disjoint s u ∧ Disjoint t u :=  disjoint_sup_left


@[simp]
theorem disjoint_union_right [DecidableEq α] {s t u : Multiset α} :
    Disjoint s (t ∪ u) ↔ Disjoint s t ∧ Disjoint s u := disjoint_sup_right


theorem add_eq_union_iff_disjoint [DecidableEq α] {s t : Multiset α} :
    s + t = s ∪ t ↔ Disjoint s t := by
  simp_rw [← inter_eq_zero_iff_disjoint, ext, count_add, count_union, count_inter, count_zero,
    Nat.min_eq_zero_iff, Nat.add_eq_max_iff]


lemma add_eq_union_left_of_le [DecidableEq α] {s t u : Multiset α} (h : t ≤ s) :
    u + s = u ∪ t ↔ Disjoint u s ∧ s = t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t u : Multiset α
    h : LE.le t s
    ⊢ Iff (Eq (HAdd.hAdd u s) (Union.union u t)) (And (Disjoint u s) (Eq s t))
  -/
  rw [← add_eq_union_iff_disjoint]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t u : Multiset α
    h : LE.le t s
    ⊢ Iff (Eq (HAdd.hAdd u s) (Union.union u t)) (And (Eq (HAdd.hAdd u s) (Union.u …
  -/
  refine ⟨fun h0 ↦ ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      s t u : Multiset α
      h : LE.le t s
      h0 : Eq (HAdd.hAdd u s) (Union.union u t)
      ⊢ And (Eq (HAdd.hAdd u s) (Union.union u s)) (Eq s t)
    -/
  · rw [and_iff_right_of_imp]
      /-
        case refine_1
        α : Type u_1
        inst✝ : DecidableEq α
        s t u : Multiset α
        h : LE.le t s
        h0 : Eq (HAdd.hAdd u s) (Union.union u t)
        ⊢ Eq s t
      -/
    · exact (le_of_add_le_add_left <| h0.trans_le <| union_le_add u t).antisymm h
      /-
        🎉 no goals
      -/
      /-
        case refine_1
        α : Type u_1
        inst✝ : DecidableEq α
        s t u : Multiset α
        h : LE.le t s
        h0 : Eq (HAdd.hAdd u s) (Union.union u t)
        ⊢ Eq s t → Eq (HAdd.hAdd u s) (Union.union u s)
      -/
    · rintro rfl
      /-
        case refine_1
        α : Type u_1
        inst✝ : DecidableEq α
        s u : Multiset α
        h : LE.le s s
        h0 : Eq (HAdd.hAdd u s) (Union.union u s)
        ⊢ Eq (HAdd.hAdd u s) (Union.union u s)
      -/
      exact h0
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : DecidableEq α
      s t u : Multiset α
      h : LE.le t s
      ⊢ And (Eq (HAdd.hAdd u s) (Union.union u s)) (Eq s t) → Eq (HAdd.hAdd u s) (Un …
    -/
  · rintro ⟨h0, rfl⟩
    /-
      case refine_2.intro
      α : Type u_1
      inst✝ : DecidableEq α
      s u : Multiset α
      h0 : Eq (HAdd.hAdd u s) (Union.union u s)
      h : LE.le s s
      ⊢ Eq (HAdd.hAdd u s) (Union.union u s)
    -/
    exact h0
    /-
      🎉 no goals
    -/


lemma add_eq_union_right_of_le [DecidableEq α] {x y z : Multiset α} (h : z ≤ y) :
    x + y = x ∪ z ↔ y = z ∧ Disjoint x y := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    x y z : Multiset α
    h : LE.le z y
    ⊢ Iff (Eq (HAdd.hAdd x y) (Union.union x z)) (And (Eq y z) (Disjoint x y))
  -/
  simpa only [and_comm] using add_eq_union_left_of_le h
  /-
    🎉 no goals
  -/


theorem disjoint_map_map {f : α → γ} {g : β → γ} {s : Multiset α} {t : Multiset β} :
    Disjoint (s.map f) (t.map g) ↔ ∀ a ∈ s, ∀ b ∈ t, f a ≠ g b := by
  /-
    α : Type u_1
    β : Type v
    γ : Type u_2
    f : α → γ
    g : β → γ
    s : Multiset α
    t : Multiset β
    ⊢ Iff (Disjoint (Multiset.map f s) (Multiset.map g t)) (∀ (a : α), Membership. …
  -/
  simp [disjoint_iff_ne]
  /-
    🎉 no goals
  -/


/-- `Pairwise r m` states that there exists a list of the elements s.t. `r` holds pairwise on this
list. -/
def Pairwise (r : α → α → Prop) (m : Multiset α) : Prop :=
  ∃ l : List α, m = l ∧ l.Pairwise r


@[simp]
theorem pairwise_zero (r : α → α → Prop) : Multiset.Pairwise r 0 :=
  ⟨[], rfl, List.Pairwise.nil⟩


theorem pairwise_coe_iff {r : α → α → Prop} {l : List α} :
    Multiset.Pairwise r l ↔ ∃ l' : List α, l ~ l' ∧ l'.Pairwise r :=
                     /-
                       α : Type u_1
                       r : α → α → Prop
                       l : List α
                       ⊢ ∀ (a : List α), Iff (And (Eq ↑l ↑a) (List.Pairwise r a)) (And (l.Perm a) (Li …
                     -/
  exists_congr <| by simp
                     /-
                       🎉 no goals
                     -/


theorem pairwise_coe_iff_pairwise {r : α → α → Prop} (hr : Symmetric r) {l : List α} :
    Multiset.Pairwise r l ↔ l.Pairwise r :=
  Iff.intro (fun ⟨_l', Eq, h⟩ => ((Quotient.exact Eq).pairwise_iff @hr).2 h) fun h => ⟨l, rfl, h⟩


theorem map_set_pairwise {f : α → β} {r : β → β → Prop} {m : Multiset α}
    (h : { a | a ∈ m }.Pairwise fun a₁ a₂ => r (f a₁) (f a₂)) : { b | b ∈ m.map f }.Pairwise r :=
  fun b₁ h₁ b₂ h₂ hn => by
    /-
      α : Type u_1
      β : Type v
      f : α → β
      r : β → β → Prop
      m : Multiset α
      h : (setOf fun a => Membership.mem m a).Pairwise fun a₁ a₂ => r (f a₁) (f a₂)
      b₁ : β
      h₁ : Membership.mem (setOf fun b => Membership.mem (Multiset.map f m) b) b₁
      b₂ : β
      h₂ : Membership.mem (setOf fun b => Membership.mem (Multiset.map f m) b) b₂
      hn : Ne b₁ b₂
      ⊢ r b₁ b₂
    -/
    obtain ⟨⟨a₁, H₁, rfl⟩, a₂, H₂, rfl⟩ := Multiset.mem_map.1 h₁, Multiset.mem_map.1 h₂
    /-
      case intro.intro.intro.intro
      α : Type u_1
      β : Type v
      f : α → β
      r : β → β → Prop
      m : Multiset α
      h : (setOf fun a => Membership.mem m a).Pairwise fun a₁ a₂ => r (f a₁) (f a₂)
      a₁ : α
      H₁ : Membership.mem m a₁
      h₁ : Membership.mem (setOf fun b => Membership.mem (Multiset.map f m) b) (f a₁)
      a₂ : α
      H₂ : Membership.mem m a₂
      h₂ : Membership.mem (setOf fun b => Membership.mem (Multiset.map f m) b) (f a₂)
      hn : Ne (f a₁) (f a₂)
      ⊢ r (f a₁) (f a₂)
    -/
    exact h H₁ H₂ (mt (congr_arg f) hn)
    /-
      🎉 no goals
    -/


/-- Given a proof `hp` that there exists a unique `a ∈ l` such that `p a`, `chooseX p l hp` returns
that `a` together with proofs of `a ∈ l` and `p a`. -/
def chooseX : ∀ _hp : ∃! a, a ∈ l ∧ p a, { a // a ∈ l ∧ p a } :=
  Quotient.recOn l (fun l' ex_unique => List.chooseX p l' (ExistsUnique.exists ex_unique))
    (by
      /-
        α : Type u_1
        β : Type v
        γ : Type u_2
        p : α → Prop
        inst✝ : DecidablePred p
        l : Multiset α
        ⊢ ∀ (a b : List α) (p_1 : HasEquiv.Equiv a b), Eq (Eq.ndrec (motive := fun x = …
      -/
      intros a b _
      /-
        α : Type u_1
        β : Type v
        γ : Type u_2
        p : α → Prop
        inst✝ : DecidablePred p
        l : Multiset α
        a b : List α
        p✝ : HasEquiv.Equiv a b
        ⊢ Eq (Eq.ndrec (motive := fun x => (ExistsUnique fun a => And (Membership.mem  …
      -/
      funext hp
      suffices all_equal : ∀ x y : { t // t ∈ b ∧ p t }, x = y by
        apply all_equal
      /-
        case h
        α : Type u_1
        β : Type v
        γ : Type u_2
        p : α → Prop
        inst✝ : DecidablePred p
        l : Multiset α
        a b : List α
        p✝ : HasEquiv.Equiv a b
        hp : ExistsUnique fun a => And (Membership.mem (Quotient.mk (List.isSetoid α)  …
        ⊢ ∀ (x y : Subtype fun t => And (Membership.mem b t) (p t)), Eq x y
      -/
      rintro ⟨x, px⟩ ⟨y, py⟩
      /-
        case h.mk.mk
        α : Type u_1
        β : Type v
        γ : Type u_2
        p : α → Prop
        inst✝ : DecidablePred p
        l : Multiset α
        a b : List α
        p✝ : HasEquiv.Equiv a b
        hp : ExistsUnique fun a => And (Membership.mem (Quotient.mk (List.isSetoid α)  …
        x : α
        px : And (Membership.mem b x) (p x)
        y : α
        py : And (Membership.mem b y) (p y)
        ⊢ Eq ⟨x, px⟩ ⟨y, py⟩
      -/
      rcases hp with ⟨z, ⟨_z_mem_l, _pz⟩, z_unique⟩
      /-
        case h.mk.mk.intro.intro.intro
        α : Type u_1
        β : Type v
        γ : Type u_2
        p : α → Prop
        inst✝ : DecidablePred p
        l : Multiset α
        a b : List α
        p✝ : HasEquiv.Equiv a b
        x : α
        px : And (Membership.mem b x) (p x)
        y : α
        py : And (Membership.mem b y) (p y)
        z : α
        z_unique : ∀ (y : α), (fun a => And (Membership.mem (Quotient.mk (List.isSetoi …
        _z_mem_l : Membership.mem (Quotient.mk (List.isSetoid α) b) z
        _pz : p z
        ⊢ Eq ⟨x, px⟩ ⟨y, py⟩
      -/
      congr
      calc
        x = z := z_unique x px
        _ = y := (z_unique y py).symm
        )


/-- Given a proof `hp` that there exists a unique `a ∈ l` such that `p a`, `choose p l hp` returns
that `a`. -/
def choose (hp : ∃! a, a ∈ l ∧ p a) : α :=
  chooseX p l hp


theorem choose_spec (hp : ∃! a, a ∈ l ∧ p a) : choose p l hp ∈ l ∧ p (choose p l hp) :=
  (chooseX p l hp).property


theorem choose_mem (hp : ∃! a, a ∈ l ∧ p a) : choose p l hp ∈ l :=
  (choose_spec _ _ _).1


theorem choose_property (hp : ∃! a, a ∈ l ∧ p a) : p (choose p l hp) :=
  (choose_spec _ _ _).2


/-- The equivalence between lists and multisets of a subsingleton type. -/
def subsingletonEquiv [Subsingleton α] : List α ≃ Multiset α where
  toFun := ofList
  invFun :=
    (Quot.lift id) fun (a b : List α) (h : a ~ b) =>
      (List.ext_get h.length_eq) fun _ _ _ => Subsingleton.elim _ _
  left_inv _ := rfl
  right_inv m := Quot.inductionOn m fun _ => rfl


@[simp]
theorem coe_subsingletonEquiv [Subsingleton α] :
    (subsingletonEquiv α : List α → Multiset α) = ofList :=
  rfl


